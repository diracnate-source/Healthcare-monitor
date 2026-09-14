# -*- coding: utf-8 -*-
"""
fusion_model.py

Fusion(v4) 모델 래퍼. model.py의 RiskModel과 같은 챔피언-챌린저 원칙을
따르되, 승격 기준이 하나 더 있다 — Fusion은 legacy(v2)뿐 아니라
**v3(학습모델) 단독보다도** 유의미하게 나아야 한다. 그렇지 않으면
"AI 예측에 양자·개인화 정보를 얹는" 이 구조 자체가 실익이 없다는
뜻이므로 승격 후보에서 제외한다.

승격 조건 (전부 만족해야 is_reliable() == True)
------------------------------------------------
1. 학습 샘플 수 >= MIN_TRAINING_SAMPLES
2. legacy(v2) 대비 AUC +0.02 이상 (is_validated_better_than_legacy)
3. v3(학습모델) 단독 대비 AUC +0.02 이상 (is_validated_better_than_v3)
4. 사람의 명시적 승인 (is_approved) — approve_model.py와 동일한
   승인 스크립트를 그대로 재사용한다(모델 폴더 구조가 동일하므로).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .fusion_features import FUSION_FEATURE_COLUMNS

MIN_FUSION_TRAINING_SAMPLES = 150
IMPROVEMENT_THRESHOLD = 0.02  # model.py와 동일한 기준을 그대로 사용


def _build_fusion_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    ])


@dataclass
class FusionTrainingReport:
    n_samples: int
    cv_auc_mean: float
    cv_auc_std: float
    legacy_auc: Optional[float]
    v3_alone_auc: Optional[float]
    feature_importances: dict

    def summary(self) -> str:
        lines = [
            f"학습 샘플 수: {self.n_samples}",
            f"Fusion 교차검증 AUC: {self.cv_auc_mean:.3f} (+/- {self.cv_auc_std:.3f})",
        ]
        if self.legacy_auc is not None:
            diff = self.cv_auc_mean - self.legacy_auc
            lines.append(f"legacy AUC: {self.legacy_auc:.3f}  (Fusion 차이 {diff:+.3f})")
        if self.v3_alone_auc is not None:
            diff_v3 = self.cv_auc_mean - self.v3_alone_auc
            lines.append(f"v3 단독 AUC: {self.v3_alone_auc:.3f}  (Fusion 차이 {diff_v3:+.3f})")
            if diff_v3 > IMPROVEMENT_THRESHOLD:
                lines.append("[판정] Fusion이 v3 단독보다 유의미하게 우수 — 승인 검토 대상")
            else:
                lines.append("[판정] Fusion이 v3 단독 대비 유의미한 개선을 보이지 않음 — 승격 보류 권장")
        return "\n".join(lines)


class FusionModel:
    """AI(v3) 예측 + QAOA 양자경로 + 개인화 정보를 입력으로 받는 메타모델."""

    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.calibrated: Optional[CalibratedClassifierCV] = None
        self.trained_on_n: int = 0
        self.is_validated_better_than_legacy: bool = False
        self.is_validated_better_than_v3: bool = False
        # model.py의 RiskModel과 동일한 원칙: 통계적 조건을 만족해도
        # 사람이 명시적으로 승인하기 전까지 절대 True가 되지 않는다.
        self.is_approved: bool = False
        self.approved_by: Optional[str] = None
        self.approved_at: Optional[str] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, calibrate: bool = True) -> None:
        X = X[FUSION_FEATURE_COLUMNS]
        pipeline = _build_fusion_pipeline()
        pipeline.fit(X, y)
        self.pipeline = pipeline
        self.trained_on_n = len(y)

        if calibrate:
            calibrated = CalibratedClassifierCV(_build_fusion_pipeline(), method="isotonic", cv=5)
            calibrated.fit(X, y)
            self.calibrated = calibrated

    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        X = X[FUSION_FEATURE_COLUMNS]
        estimator = self.calibrated if self.calibrated is not None else self.pipeline
        if estimator is None:
            raise RuntimeError("Fusion 모델이 아직 학습되지 않았습니다.")
        return estimator.predict_proba(X)[:, 1] * 100.0

    def is_reliable(self) -> bool:
        """네 가지 조건(샘플 수, legacy 대비 우위, v3 단독 대비 우위,
        사람의 승인)을 전부 만족해야 True. 하나라도 빠지면 Fusion은
        사용되지 않고 v3(또는 legacy)로 폴백해야 한다 — 이 판단은
        app_py_adapter.py 쪽에서 이 값을 확인해 처리한다.
        """
        return (
            self.pipeline is not None
            and self.trained_on_n >= MIN_FUSION_TRAINING_SAMPLES
            and self.is_validated_better_than_legacy
            and self.is_validated_better_than_v3
            and self.is_approved
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path / "pipeline.joblib")
        if self.calibrated is not None:
            joblib.dump(self.calibrated, path / "calibrated.joblib")
        meta = {
            "trained_on_n": self.trained_on_n,
            "is_validated_better_than_legacy": self.is_validated_better_than_legacy,
            "is_validated_better_than_v3": self.is_validated_better_than_v3,
            "is_approved": self.is_approved,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "feature_columns": FUSION_FEATURE_COLUMNS,
            "model_type": "fusion_v4",
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FusionModel":
        path = Path(path)
        model = cls()
        model.pipeline = joblib.load(path / "pipeline.joblib")
        calibrated_path = path / "calibrated.joblib"
        if calibrated_path.exists():
            model.calibrated = joblib.load(calibrated_path)
        meta = json.loads((path / "meta.json").read_text())
        model.trained_on_n = meta["trained_on_n"]
        model.is_validated_better_than_legacy = meta["is_validated_better_than_legacy"]
        model.is_validated_better_than_v3 = meta.get("is_validated_better_than_v3", False)
        model.is_approved = meta.get("is_approved", False)
        model.approved_by = meta.get("approved_by")
        model.approved_at = meta.get("approved_at")
        return model
