# -*- coding: utf-8 -*-
"""
fusion_features.py

그림에서 말한 "통합 알고리즘"의 실제 구현:

    AI(v3) 예측값 + QAOA 최적 가중치 + 개인 baseline + 결과 일관성
                            ↓
                    Fusion 메타모델 (학습됨)
                            ↓
                        최종 위험도

중요한 설계 원칙
----------------
Fusion은 "AI와 양자를 어떻게 섞을지 사람이 정한 공식"이 아니다. 아래
피처들을 입력으로 받아 라벨 데이터로 학습되는 하나의 작은 모델
(fusion_model.py의 FusionModel)이다. 즉 "이 4가지 정보를 합치는 게
실제로 예측력을 높이는가"는 우리가 가정하지 않고, train_fusion.py가
legacy·v3-단독 대비 AUC로 직접 검증한다. 도움이 안 되면 승격되지
않는다 — 지금까지의 챔피언-챌린저 원칙과 동일하다.

피처 구성
----------
1. ai_pred_proba        : v3(RiskModel)가 예측한 위험 확률(0~1).
                          이미 features.py의 FEATURE_COLUMNS로 계산된
                          것을 그대로 재사용한다 — Fusion이 처음부터
                          다시 6개 지표를 보는 게 아니라, v3의 '결론'
                          위에 얹는 stacking 구조다.
2. {metric}_qaoa_boosted : QAOA가 이 지표의 가중치를 부스트했는지(0/1).
                          quantum_optimizer.compute_quantum_optimized_score()의
                          boosted_metrics 결과를 그대로 원-핫으로 편다.
3. qaoa_combined_z       : QAOA 가중결합으로 나온 combined_Z (양자 경로의
                          '위험도 방향' 요약값).
4. qaoa_top_probability  : QAOA가 최종 해에 도달한 확신도
                          (top_bitstring_probability) — 회로가 특정
                          조합에 얼마나 뚜렷하게 수렴했는지.
5. qaoa_matched_true_optimum : QAOA 자체가 스스로 진짜 최적해를
                          찾았는지(0/1) — 못 찾아 전수조사로 보정된
                          세션인지 여부도 정보가 될 수 있다.
6. ai_qaoa_score_diff    : |AI risk_score - QAOA risk_score| (0~100 스케일
                          기준). '결과 일관성'을 모델이 스스로 재발견할
                          필요 없이 명시적으로 넣어준다.
7. ai_qaoa_tier_agree    : AI와 QAOA의 3단계 판정이 일치하는지(0/1).
8. n_personal, population_ready_frac, {metric}_personal_z
                          : 개인 baseline 정보. features.py가 이미
                          계산한 것을 그대로 재사용한다(중복 계산 없음).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .legacy_scoring import METRIC_NAMES
from .features import FEATURE_COLUMNS as V3_FEATURE_COLUMNS

FUSION_FEATURE_COLUMNS = (
    ["ai_pred_proba"]
    + [f"{m}_qaoa_boosted" for m in METRIC_NAMES]
    + ["qaoa_combined_z", "qaoa_top_probability", "qaoa_matched_true_optimum"]
    + ["ai_qaoa_score_diff", "ai_qaoa_tier_agree"]
    + ["n_personal", "population_ready_frac"]
    + [col for col in V3_FEATURE_COLUMNS if col.endswith("_personal_z")]
)

_TIER_ORDER = {"양호": 0, "주의": 1, "확인 권장": 2}


def build_fusion_feature_row(
    ai_pred_proba: float,
    ai_risk_score: float,
    ai_tier: str,
    quantum_result: Optional[dict],
    v3_feature_row: Dict[str, float],
) -> Dict[str, float]:
    """세션 1건에 대한 Fusion 입력 피처 행을 만든다.

    quantum_result가 None이면(부트스트랩 초기 상태에서 양자 경로 계산
    자체가 안 된 세션 등) QAOA 관련 피처는 전부 NaN으로 채운다 —
    임의로 0을 채우면 '부스트 없음'과 '계산 자체가 안 됨'이 뒤섞여
    모델이 잘못된 패턴을 학습할 수 있어서, 결측은 결측 그대로 둔다.
    """
    row: Dict[str, float] = {"ai_pred_proba": ai_pred_proba}

    if quantum_result is None:
        for m in METRIC_NAMES:
            row[f"{m}_qaoa_boosted"] = np.nan
        row["qaoa_combined_z"] = np.nan
        row["qaoa_top_probability"] = np.nan
        row["qaoa_matched_true_optimum"] = np.nan
        row["ai_qaoa_score_diff"] = np.nan
        row["ai_qaoa_tier_agree"] = np.nan
    else:
        boosted = set(quantum_result.get("boosted_metrics", []))
        for m in METRIC_NAMES:
            row[f"{m}_qaoa_boosted"] = 1.0 if m in boosted else 0.0

        row["qaoa_combined_z"] = float(quantum_result.get("combined_z", np.nan))
        row["qaoa_top_probability"] = float(quantum_result.get("top_bitstring_probability", np.nan))
        row["qaoa_matched_true_optimum"] = float(
            bool(quantum_result.get("qaoa_matched_true_optimum", False))
        )

        q_risk_score = quantum_result.get("risk_score")
        if q_risk_score is not None:
            row["ai_qaoa_score_diff"] = abs(ai_risk_score - q_risk_score)
        else:
            row["ai_qaoa_score_diff"] = np.nan

        q_tier = quantum_result.get("tier")
        if q_tier is not None and ai_tier is not None:
            row["ai_qaoa_tier_agree"] = 1.0 if q_tier == ai_tier else 0.0
        else:
            row["ai_qaoa_tier_agree"] = np.nan

    # 개인화 정보는 v3 피처에서 그대로 재사용 (중복 계산 방지)
    row["n_personal"] = v3_feature_row.get("n_personal", np.nan)
    row["population_ready_frac"] = v3_feature_row.get("population_ready_frac", np.nan)
    for m in METRIC_NAMES:
        key = f"{m}_personal_z"
        row[key] = v3_feature_row.get(key, np.nan)

    return row


def build_fusion_feature_frame(rows: list[Dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in FUSION_FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[FUSION_FEATURE_COLUMNS]
