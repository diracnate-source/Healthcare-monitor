# -*- coding: utf-8 -*-
"""
uncertainty.py

위험도 점수(risk_score)에 신뢰구간을 추가하는 모듈.

왜 필요한가
------------
지금까지 시스템은 "확인 권장" 같은 판정과 "66.5점" 같은 딱 떨어지는 숫자만
보여줬다. 그런데 이 66.5점은 유한한 표본(모집단 n, 개인 n_personal)으로
추정한 평균·표준편차 위에서 계산된 값이라, 표본이 적을수록 그 추정
자체가 불확실하다. 의료 판단에 참고하는 수치라면 "이 점수가 얼마나
믿을 만한 추정인지"도 함께 제시해야 한다.

방법: 파라메트릭 부트스트랩
----------------------------
실제 관측값(x, 이번 세션의 측정치)은 고정한 채, 그 값을 평가하는 기준인
모집단/개인 평균·표준편차를 "우리가 모르는 진짜 값 주변의 불확실한
추정치"로 보고 반복적으로 다시 뽑는다(resampling). 그때마다 전체
파이프라인(EB 결합 → 단측 Z-score → 가중결합 → 지수 스쿼싱)을 그대로
다시 계산해서 risk_score 분포를 만들고, 그 분포의 2.5~97.5 백분위수를
95% 신뢰구간으로 보고한다.

- 평균의 표본오차: SE_mean = std / sqrt(n)  (표준적인 통계 공식)
- 표준편차의 표본오차: SE_std ≈ std / sqrt(2*(n-1))  (정규분포 가정 하의 근사식)

표본(n 또는 n_personal)이 적을수록 신뢰구간이 넓어지고, 표본이 쌓일수록
좁아진다 — 즉 "표본이 적을 때는 점수를 덜 확신해야 한다"는 사실이
숫자로 그대로 드러난다.

주의
----
- 이 신뢰구간은 '모집단/개인 기준 추정치의 불확실성'만 반영한다.
  얼굴 랜드마크 검출 자체의 측정 오차, 카메라·조명 조건에 따른 노이즈는
  포함하지 않는다 — 즉 실제 총 불확실성의 하한선(underestimate)일 수
  있다는 점을 명시해야 한다.
- 데이터기반 모드(data_driven)에서만 통계적으로 의미가 있다. 부트스트랩
  모드(고정 계수)에는 애초에 확률적으로 추정된 파라미터가 없으므로
  신뢰구간을 계산하지 않고 그 사실을 명시한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .legacy_scoring import (
    METRIC_NAMES,
    METRIC_WEIGHTS,
    SQUASH_LAMBDA,
    EMPIRICAL_BAYES_K,
    MIN_PERSONAL_SESSIONS_FOR_VARIANCE,
    PersonalStats,
    WelfordAccumulator,
    _tier_from_z,
)

N_BOOTSTRAP_SAMPLES = 2000
_RNG_SEED = 20260910  # 재현성을 위해 고정 — 같은 입력이면 항상 같은 신뢰구간


@dataclass
class RiskScoreWithUncertainty:
    point_estimate: float           # 원래 방식 그대로의 risk_score (변경 없음)
    tier: str
    ci_lower: Optional[float]       # 95% 신뢰구간 하한
    ci_upper: Optional[float]       # 95% 신뢰구간 상한
    ci_available: bool
    note: str


def _resample_mean_std(mean: float, std: float, n: int, rng: np.random.Generator):
    """평균·표준편차 추정치 각각의 표본오차만큼 정규분포로 흔들어
    '있을 법한 진짜 모수'를 하나 뽑는다."""
    se_mean = std / math.sqrt(max(n, 1))
    se_std = std / math.sqrt(max(2 * (n - 1), 1)) if n > 1 else std

    resampled_mean = rng.normal(mean, se_mean)
    resampled_std = max(1e-6, rng.normal(std, se_std))  # 표준편차는 항상 양수
    return resampled_mean, resampled_std


def compute_risk_score_ci(
    metrics: Dict[str, Optional[float]],
    population_stats: Dict[str, WelfordAccumulator],
    personal_stats: Dict[str, PersonalStats],
    point_estimate: float,
    tier: str,
    mode: str,
    n_samples: int = N_BOOTSTRAP_SAMPLES,
) -> RiskScoreWithUncertainty:
    """data_driven 모드일 때만 실제로 신뢰구간을 계산한다. bootstrap
    모드에서는 계산할 확률적 파라미터가 없으므로 '해당 없음'을 명시해
    반환한다 — 억지로 숫자를 만들어내지 않는다.
    """
    if mode != "data_driven":
        return RiskScoreWithUncertainty(
            point_estimate=point_estimate, tier=tier,
            ci_lower=None, ci_upper=None, ci_available=False,
            note="부트스트랩(초기) 모드에서는 신뢰구간을 계산하지 않습니다 — "
                 "모집단 표본이 아직 충분하지 않아 확률적으로 추정된 파라미터가 없습니다.",
        )

    rng = np.random.default_rng(_RNG_SEED)
    sampled_scores: List[float] = []

    for _ in range(n_samples):
        weighted_z_sum = 0.0
        weight_sum = 0.0

        for name in METRIC_NAMES:
            x = metrics.get(name)
            pop = population_stats.get(name)
            pers = personal_stats.get(name)
            if x is None or pop is None or pop.std is None or pop.std <= 1e-6:
                continue

            pop_mean_s, pop_std_s = _resample_mean_std(pop.mean, pop.std, pop.n, rng)

            n_personal = pers.n_personal if pers else 0
            mu_personal = pers.mean if pers and pers.n_personal > 0 else pop.mean
            if pers and pers.n_personal > 1:
                pers_mean_s, _ = _resample_mean_std(mu_personal, pop.std, pers.n_personal, rng)
            else:
                pers_mean_s = mu_personal

            w_personal = n_personal / (n_personal + EMPIRICAL_BAYES_K)
            mu_ref_s = w_personal * pers_mean_s + (1 - w_personal) * pop_mean_s

            if (pers and pers.n_personal >= MIN_PERSONAL_SESSIONS_FOR_VARIANCE
                    and pers.std is not None and pers.std > 1e-6):
                _, pers_std_s = _resample_mean_std(pers.std, pers.std, pers.n_personal, rng)
                w_sigma = (n_personal - MIN_PERSONAL_SESSIONS_FOR_VARIANCE) / (
                    (n_personal - MIN_PERSONAL_SESSIONS_FOR_VARIANCE) + EMPIRICAL_BAYES_K
                )
                sigma_ref_s = w_sigma * pers_std_s + (1 - w_sigma) * pop_std_s
            else:
                sigma_ref_s = pop_std_s

            z = (x - mu_ref_s) / sigma_ref_s
            z_plus = max(0.0, z)

            weight = METRIC_WEIGHTS[name]
            weighted_z_sum += weight * z_plus
            weight_sum += weight

        if weight_sum == 0:
            continue

        combined_z_s = weighted_z_sum / weight_sum
        score_s = 100.0 * (1.0 - math.exp(-combined_z_s / SQUASH_LAMBDA))
        sampled_scores.append(max(0.0, min(100.0, score_s)))

    if len(sampled_scores) < n_samples * 0.5:
        return RiskScoreWithUncertainty(
            point_estimate=point_estimate, tier=tier,
            ci_lower=None, ci_upper=None, ci_available=False,
            note="신뢰구간 계산에 필요한 유효 표본이 부족합니다.",
        )

    ci_lower = float(np.percentile(sampled_scores, 2.5))
    ci_upper = float(np.percentile(sampled_scores, 97.5))

    return RiskScoreWithUncertainty(
        point_estimate=point_estimate, tier=tier,
        ci_lower=ci_lower, ci_upper=ci_upper, ci_available=True,
        note=(
            "95% 신뢰구간: 모집단·개인 평균/표준편차 추정 자체의 불확실성만 반영한 값이며, "
            "얼굴 인식 측정 오차나 촬영 환경 노이즈는 포함하지 않습니다. 표본이 늘어날수록 "
            "구간이 좁아집니다."
        ),
    )
