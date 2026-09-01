# -*- coding: utf-8 -*-
"""
legacy_scoring.py

biomarker_formulas_v2.docx 에 정의된 v2 통합 위험도 알고리즘
(부트스트랩 모드 + 데이터기반 모드 / Welford + 경험적 베이즈 + 단측 Z-score
 + 지수 스쿼싱)을 그대로 코드로 옮긴 모듈.

이 모듈은 두 가지 용도로 쓰인다.
  1) 학습된 모델(model.py의 RiskModel)이 아직 없거나(콜드스타트),
     신뢰할 수 없을 때의 안전한 폴백(fallback)
  2) 새로 학습한 모델의 성능이 실제로 v2보다 나은지 비교하는 기준선(baseline)

주의: 여기 있는 계수(LEGACY_WEIGHTS, METRIC_WEIGHTS, K, LAMBDA, TIER 경계값)는
전부 문서에 명시된 대로 "임상 라벨로 검증되지 않은 잠정값"이다. 절대 이 값들이
정답이라고 가정하고 학습 모델의 피처를 설계하지 말 것 — 학습 모델은 이 값들과
무관하게 데이터로부터 자기 자신의 가중치를 배우게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

MIN_POPULATION_N = 30
EMPIRICAL_BAYES_K = 4
SQUASH_LAMBDA = 2.0

LEGACY_WEIGHTS = {
    "expression_change": 0.6,
    "micro_movement": 0.8,
    "facial_asymmetry": 0.5,
    "blink_rate": 1.0,
    "gaze_variability": 0.4,
    # reaction_ms 는 별도 산식 (아래 _legacy_reaction_raw)
}

METRIC_WEIGHTS = {
    "expression_change": 0.8,
    "micro_movement": 0.5,
    "facial_asymmetry": 1.0,
    "blink_rate": 1.0,
    "gaze_variability": 0.8,
    "reaction_ms": 1.0,
}

METRIC_NAMES = (
    "expression_change",
    "micro_movement",
    "facial_asymmetry",
    "blink_rate",
    "gaze_variability",
    "reaction_ms",
)

# app.py 화면 표시용 한글 라벨 (원본 app.py의 METRIC_LABELS와 동일)
METRIC_LABELS = {
    "expression_change": "표정 변화",
    "micro_movement": "미세 근육 움직임",
    "facial_asymmetry": "안면 비대칭",
    "blink_rate": "눈 깜빡임 빈도(회/분)",
    "gaze_variability": "시선 이동 변동성",
    "reaction_ms": "반응속도(ms)",
}


@dataclass
class WelfordAccumulator:
    """모집단 기준 통계 (Welford's online algorithm). 지표 하나당 1개씩 운용."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> Optional[float]:
        if self.n < 2:
            return None
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> Optional[float]:
        var = self.variance
        return None if var is None else var**0.5

    @property
    def ready(self) -> bool:
        """표본 수 충분 + 표준편차가 0에 가깝지 않을 때만 '신뢰 가능'."""
        std = self.std
        return self.n >= MIN_POPULATION_N and std is not None and std > 1e-6


@dataclass
class PersonalStats:
    """사용자 1인의 지표별 러닝 평균 및 누적 세션 수."""

    n_personal: int = 0
    mean: float = 0.0

    def update(self, x: float) -> None:
        self.n_personal += 1
        self.mean += (x - self.mean) / self.n_personal


def _legacy_reaction_raw(reaction_ms: Optional[float]) -> Optional[float]:
    if reaction_ms is None:
        return None
    return max(0.0, (reaction_ms - 300.0) * 0.05)


def bootstrap_score(metrics: Dict[str, Optional[float]]) -> Dict:
    """모집단 표본 < MIN_POPULATION_N 일 때: 레거시 고정계수 선형가중합."""
    raw = {}
    for name, weight in LEGACY_WEIGHTS.items():
        value = metrics.get(name)
        raw[name] = None if value is None else value * weight
    raw["reaction_ms"] = _legacy_reaction_raw(metrics.get("reaction_ms"))

    scoreable = [v for v in raw.values() if v is not None]
    if not scoreable:
        raise ValueError("계산 가능한 지표가 하나도 없습니다.")

    mean_raw = sum(scoreable) / len(scoreable)
    risk_score = max(0.0, min(100.0, mean_raw * 1.2))
    return {
        "mode": "bootstrap",
        "risk_score": risk_score,
        "tier": _tier_from_score(risk_score),
    }


def _tier_from_score(risk_score: float) -> str:
    if risk_score < 33:
        return "양호"
    if risk_score < 66:
        return "주의"
    return "확인 권장"


def _tier_from_z(combined_z: float) -> str:
    if combined_z < 0.5:
        return "양호"
    if combined_z < 1.5:
        return "주의"
    return "확인 권장"


def data_driven_score(
    metrics: Dict[str, Optional[float]],
    population_stats: Dict[str, WelfordAccumulator],
    personal_stats: Dict[str, PersonalStats],
) -> Dict:
    """모집단 표본 >= MIN_POPULATION_N 일 때: EB 결합 + 단측 Z-score + 지수 스쿼싱."""
    weighted_z_sum = 0.0
    weight_sum = 0.0
    per_metric_detail = {}

    for name in METRIC_NAMES:
        x = metrics.get(name)
        pop = population_stats.get(name)
        pers = personal_stats.get(name)
        if x is None or pop is None or pop.std is None or pop.std <= 1e-6:
            continue

        n_personal = pers.n_personal if pers else 0
        mu_personal = pers.mean if pers and pers.n_personal > 0 else pop.mean
        w_personal = n_personal / (n_personal + EMPIRICAL_BAYES_K)

        mu_ref = w_personal * mu_personal + (1 - w_personal) * pop.mean
        sigma_ref = pop.std

        z = (x - mu_ref) / sigma_ref
        z_plus = max(0.0, z)

        weight = METRIC_WEIGHTS[name]
        weighted_z_sum += weight * z_plus
        weight_sum += weight
        per_metric_detail[name] = {"z_plus": z_plus, "mu_ref": mu_ref, "sigma_ref": sigma_ref}

    if weight_sum == 0:
        raise ValueError("Z-score를 계산할 수 있는 지표가 없습니다(모집단 통계 미준비).")

    combined_z = weighted_z_sum / weight_sum
    risk_score = 100.0 * (1.0 - pow(2.718281828459045, -combined_z / SQUASH_LAMBDA))
    risk_score = max(0.0, min(100.0, risk_score))

    return {
        "mode": "data_driven",
        "combined_z": combined_z,
        "risk_score": risk_score,
        "tier": _tier_from_z(combined_z),
        "detail": per_metric_detail,
    }


def compute_legacy_risk(
    metrics: Dict[str, Optional[float]],
    population_stats: Dict[str, WelfordAccumulator],
    personal_stats: Dict[str, PersonalStats],
) -> Dict:
    """게이트: blink_rate 모집단 표본 수(n)로 세션 전체 모드를 결정."""
    gate = population_stats.get("blink_rate")
    if gate is None or gate.n < MIN_POPULATION_N:
        return bootstrap_score(metrics)
    return data_driven_score(metrics, population_stats, personal_stats)
