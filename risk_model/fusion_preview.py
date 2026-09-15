# -*- coding: utf-8 -*-
"""
fusion_preview.py

사용자가 요청한 구조 — "6개 지표 → (AI 분석 / QAOA 분석) → 통합 알고리즘
(AI 예측 + QAOA 가중치 + 개인 baseline + 결과 일관성) → 최종 위험도" —
를 규칙 기반으로 즉시 계산하는 잠정(preview) 모듈이다.

이것과 fusion_model.py(FusionModel, v4)의 차이
--------------------------------------------------
fusion_model.py는 이 4가지 정보를 "라벨 데이터로 학습된 모델"이 스스로
결합 방식을 찾아내게 한다 — 그래서 실사용 라벨이 최소 150건 쌓이고,
legacy/v3 단독 대비 검증되고, 사람이 승인해야만 실제로 쓰인다
(FusionModel.is_reliable()).

지금은 그 조건을 만족하지 못한다(라벨 자체가 없음). 이 모듈은 그
학습된 모델이 준비되기 전까지, 같은 4가지 입력을 사람이 정한 명시적
공식으로 즉시 결합해 화면에 보여주기 위한 것이다. 학습된 결합이
아니므로, 예측력에 대한 어떠한 근거도 없다 — 순수하게 "그림에 그린
구조를 지금 당장 동작하는 형태로 보여주기 위한" 잠정 버전이다.

공식 (전부 설명 가능하고, 극단적 상황에서 안전한 방향으로 설계)
------------------------------------------------------------------
1. base_avg = 0.5 * AI위험도 + 0.5 * QAOA위험도
   — 두 경로를 동등하게 취급한 단순 평균. 어느 한쪽이 더 낫다는 근거가
     없으므로 임의로 가중치를 다르게 주지 않는다.
2. conservative = max(AI위험도, QAOA위험도)
   — 안전 방향으로 치우친 값. 두 경로 중 더 주의가 필요한 쪽.
3. personal_confidence = min(1, n_personal / MIN_PERSONAL_SESSIONS_FOR_VARIANCE)
   — "개인 baseline" 반영: legacy_scoring.py가 개인 분산을 신뢰하기
     시작하는 기준(세션 8회)을 그대로 재사용한다. 개인 이력이 적을수록
     (콜드스타트) conservative 쪽에, 충분히 쌓일수록 base_avg 쪽에
     가깝게 움직인다 — "아직 이 사람을 잘 모를 때는 보수적으로,
     알수록 두 경로를 동등하게 신뢰한다"는 원칙이다.
4. fusion_score = personal_confidence * base_avg
                  + (1 - personal_confidence) * conservative
5. consistency = 1 - |AI위험도 - QAOA위험도| / 100
   — "결과 일관성": 점수 자체에 반영하지 않고, 얼마나 신뢰할 만한
     통합인지 별도로 표시한다(0=극단적으로 다름, 1=완전히 일치).
     점수 계산에 섞지 않는 이유: 일관성이 낮다고 점수를 더 올리거나
     내리는 것은 근거 없는 임의 조정이 되기 때문이다 — 대신 "이번
     결과는 일관성이 낮으니 참고용으로만 보라"는 신호로만 쓴다.

주의
----
이 함수가 반환하는 fusion_score는 학습·검증된 예측값이 아니다. 어디까지나
"그림의 구조를 지금 동작하게 만든" 명시적 규칙이며, 화면에도 항상 이
사실을 노출해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .legacy_scoring import MIN_PERSONAL_SESSIONS_FOR_VARIANCE, _tier_from_score


@dataclass
class FusionPreviewResult:
    fusion_score: float
    tier: str
    ai_score: float
    quantum_score: Optional[float]
    base_avg: Optional[float]
    conservative: Optional[float]
    personal_confidence: float
    consistency: Optional[float]
    note: str


def compute_fusion_preview(
    ai_risk_score: float,
    quantum_result: Optional[dict],
    n_personal: int,
) -> FusionPreviewResult:
    """AI 예측 + QAOA 가중치 + 개인 baseline + 결과 일관성을 규칙
    기반으로 결합한다. quantum_result가 None이면(양자 경로 미계산)
    AI 단독 결과를 그대로 fusion_score로 반환한다 — 값을 지어내지
    않는다.
    """
    personal_confidence = min(1.0, n_personal / MIN_PERSONAL_SESSIONS_FOR_VARIANCE)

    if quantum_result is None:
        return FusionPreviewResult(
            fusion_score=ai_risk_score,
            tier=_tier_from_score(ai_risk_score),
            ai_score=ai_risk_score,
            quantum_score=None,
            base_avg=None,
            conservative=None,
            personal_confidence=personal_confidence,
            consistency=None,
            note=(
                "양자 경로가 아직 계산되지 않아 AI 경로 결과를 그대로 "
                "통합 결과로 사용했습니다."
            ),
        )

    q_score = float(quantum_result["risk_score"])
    base_avg = 0.5 * ai_risk_score + 0.5 * q_score
    conservative = max(ai_risk_score, q_score)

    fusion_score = personal_confidence * base_avg + (1 - personal_confidence) * conservative
    fusion_score = max(0.0, min(100.0, fusion_score))

    consistency = 1.0 - abs(ai_risk_score - q_score) / 100.0

    return FusionPreviewResult(
        fusion_score=fusion_score,
        tier=_tier_from_score(fusion_score),
        ai_score=ai_risk_score,
        quantum_score=q_score,
        base_avg=base_avg,
        conservative=conservative,
        personal_confidence=personal_confidence,
        consistency=consistency,
        note=(
            "규칙 기반 잠정 결합(학습된 모델 아님): AI·QAOA 단순평균과 "
            "'더 주의가 필요한 쪽' 사이를, 개인 기록이 쌓인 정도에 따라 "
            "가중 결합했습니다. 개인 기록이 많을수록 단순평균에, "
            "적을수록(콜드스타트) 보수적인 쪽에 가깝게 계산됩니다."
        ),
    )
