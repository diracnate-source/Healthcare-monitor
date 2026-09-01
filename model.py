# -*- coding: utf-8 -*-
"""
model.py

"6개 지표 + 인구통계 + 개인/모집단 통계 -> 학습된 모델 -> risk_score" 의
학습/추론을 담당.

지금 단계(라벨 데이터가 적은 PoC~초기 서비스)에 맞춘 설계:

  1. 베이스라인은 정규화 로지스틱 회귀(L2). 해석 가능하고, 적은 데이터에서도
     비교적 안정적이며, "어떤 피처가 위험도에 얼마나 기여했는가"를 계수로
     바로 설명할 수 있다.
  2. 라벨이 충분히 쌓이면(권장 최소치는 train.py 참고) Gradient Boosting으로
     전환 옵션 제공 — 지표 간 비선형 상호작용과 결측치 자체 처리가 강점.
  3. 확률 보정(calibration): risk_score를 "확률처럼" 해석하게 하려면 모델의
     원출력이 실제 위험 비율과 맞아야 한다. CalibratedClassifierCV로 보정.
  4. 콜드스타트/모델 미신뢰 상황에서는 legacy_scoring.py 로 자동 폴백한다.
     학습 모델이 "확실히 legacy보다 낫다"고 검증되기 전까지, 이 폴백을
     기본값으로 유지할 것을 권장한다 (README 참고).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS
from .legacy_scoring import PersonalStats, WelfordAccumulator, compute_legacy_risk

# 이 모델이 실전에 투입되려면 최소 이만큼의 라벨된 세션이 있어야 한다고 보는
# 임시 기준. 절대적 정답은 아니며, 실제로는 학습곡선(learning curve)을 그려
# 검증오차가 안정화되는 지점을 확인해서 조정해야 한다.
MIN_TRAINING_SAMPLES = 150


def _build_pipeline() -> Pipeline:
    """로지스틱 회귀 베이스라인 파이프라인.

    - SimpleImputer: 결측을 중앙값으로 채우되, features.py에서 이미 만든
      '{metric}_missing' 플래그가 있어 정보 손실을 보완한다.
    - StandardScaler: 스케일이 서로 다른 원시지표/z값/나이를 정규화.
    - LogisticRegression: L2 정규화, 클래스 불균형 대비 class_weight='balanced'.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
        ]
    )


@dataclass
class TrainingReport:
    n_samples: int
    cv_auc_mean: float
    cv_auc_std: float
    cv_sensitivity_mean: float
    cv_specificity_mean: float
    legacy_auc: Optional[float]  # 같은 데이터에서 legacy v2 방식의 AUC (비교용)
    feature_importances: Dict[str, float]

    def summary(self) -> str:
        lines = [
            f"학습 샘플 수: {self.n_samples}",
            f"교차검증 AUC: {self.cv_auc_mean:.3f} (+/- {self.cv_auc_std:.3f})",
            f"교차검증 민감도(재현율): {self.cv_sensitivity_mean:.3f}",
            f"교차검증 특이도: {self.cv_specificity_mean:.3f}",
        ]
        if self.legacy_auc is not None:
            lines.append(f"[비교] 동일 데이터에서 legacy v2 AUC: {self.legacy_auc:.3f}")
            diff = self.cv_auc_mean - self.legacy_auc
            verdict = "학습모델이 더 우수" if diff > 0.02 else (
                "유의미한 차이 없음 - 아직 legacy 폴백 유지 권장" if abs(diff) <= 0.02
                else "legacy가 더 우수 - 학습모델 배포 보류 권장"
            )
            lines.append(f"[판정] {verdict} (AUC 차이 {diff:+.3f})")
        return "\n".join(lines)


class RiskModel:
    """학습된 위험도 모델 + 콜드스타트 폴백을 함께 관리하는 래퍼."""

    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.calibrated: Optional[CalibratedClassifierCV] = None
        self.trained_on_n: int = 0
        self.is_validated_better_than_legacy: bool = False

    # ---------- 학습 ----------
    def fit(self, X: pd.DataFrame, y: np.ndarray, calibrate: bool = True) -> None:
        X = X[FEATURE_COLUMNS]
        pipeline = _build_pipeline()
        pipeline.fit(X, y)
        self.pipeline = pipeline
        self.trained_on_n = len(y)

        if calibrate:
            calibrated = CalibratedClassifierCV(_build_pipeline(), method="isotonic", cv=5)
            calibrated.fit(X, y)
            self.calibrated = calibrated

    # ---------- 추론 ----------
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """0~100 스케일 risk_score 반환 (모델의 위험확률 * 100)."""
        X = X[FEATURE_COLUMNS]
        estimator = self.calibrated if self.calibrated is not None else self.pipeline
        if estimator is None:
            raise RuntimeError("모델이 아직 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        proba = estimator.predict_proba(X)[:, 1]
        return proba * 100.0

    def is_reliable(self) -> bool:
        """이 모델을 실제 서비스에서 legacy 대신 쓸 만큼 신뢰할 수 있는가.

        단순히 '학습됐다'가 아니라, (a) 충분한 샘플로 학습되었고
        (b) 별도 검증에서 legacy보다 낫다고 확인된 경우에만 True.
        둘 중 하나라도 아니면 legacy로 폴백하는 것이 안전하다.
        """
        return (
            self.pipeline is not None
            and self.trained_on_n >= MIN_TRAINING_SAMPLES
            and self.is_validated_better_than_legacy
        )

    # ---------- 저장/로드 ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path / "pipeline.joblib")
        if self.calibrated is not None:
            joblib.dump(self.calibrated, path / "calibrated.joblib")
        meta = {
            "trained_on_n": self.trained_on_n,
            "is_validated_better_than_legacy": self.is_validated_better_than_legacy,
            "feature_columns": FEATURE_COLUMNS,
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RiskModel":
        path = Path(path)
        model = cls()
        model.pipeline = joblib.load(path / "pipeline.joblib")
        calibrated_path = path / "calibrated.joblib"
        if calibrated_path.exists():
            model.calibrated = joblib.load(calibrated_path)
        meta = json.loads((path / "meta.json").read_text())
        model.trained_on_n = meta["trained_on_n"]
        model.is_validated_better_than_legacy = meta["is_validated_better_than_legacy"]
        return model


def score_with_fallback(
    model: Optional[RiskModel],
    feature_row: pd.DataFrame,
    raw_metrics: Dict[str, Optional[float]],
    population_stats: Dict[str, WelfordAccumulator],
    personal_stats: Dict[str, PersonalStats],
) -> Dict:
    """서비스 레벨 진입점.

    학습 모델이 없거나, 있어도 아직 legacy보다 낫다고 검증되지 않았다면
    (is_reliable() == False) 자동으로 legacy_scoring.compute_legacy_risk 로
    폴백한다. 이 함수 하나만 호출하면 되도록 만들어, 운영 코드가 "지금
    어떤 모드인지"를 신경 쓰지 않아도 된다.
    """
    if model is not None and model.is_reliable():
        risk_score = float(model.predict_risk_score(feature_row)[0])
        return {
            "mode": "learned_model",
            "risk_score": risk_score,
            "tier": _tier_from_score(risk_score),
        }

    legacy_result = compute_legacy_risk(raw_metrics, population_stats, personal_stats)
    legacy_result["mode"] = f"fallback_{legacy_result['mode']}"
    return legacy_result


def _tier_from_score(risk_score: float) -> str:
    if risk_score < 33:
        return "양호"
    if risk_score < 66:
        return "주의"
    return "확인 권장"
