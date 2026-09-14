# -*- coding: utf-8 -*-
"""
train_fusion.py

Fusion(v4) 모델을 학습하고, legacy(v2)뿐 아니라 v3(학습모델) 단독
대비로도 AUC를 비교 검증하는 CLI.

입력 데이터 요구사항 (중요)
----------------------------
이 스크립트는 "이미 AI(v3) 예측과 QAOA 양자경로 결과가 함께 기록된"
라벨 데이터를 전제로 한다. 즉 다음이 선행되어야 한다.

  1. dual_path.py가 운영 중 쌓아온 dual_path_log.jsonl (AI 예측 +
     QAOA 결과가 세션마다 기록됨)
  2. 그 세션들 중 일부에 대해 사후적으로 확보된 임상 라벨(진단/MMSE 등)

이 두 데이터를 user_id·시각 기준으로 조인해서, 아래 컬럼을 가진 CSV로
준비해야 한다.

  sample_id, user_id,
  expression_change, micro_movement, facial_asymmetry, blink_rate,
  gaze_variability, reaction_ms,
  age, sex, education_years, n_personal,
  qaoa_boosted_metrics (예: "facial_asymmetry|reaction_ms" 파이프 구분),
  qaoa_combined_z, qaoa_top_probability, qaoa_matched_true_optimum (0/1),
  ai_risk_score, ai_tier,
  label

이 조인 파이프라인 자체(dual_path_log.jsonl + 라벨 소스 → CSV)는 아직
구현되어 있지 않다 — 라벨 확보 방식이 병원·기관마다 다를 것이므로,
지금은 CSV 스키마까지만 정의해두고 실제 조인 스크립트는 라벨 데이터가
실제로 확보되는 시점에 맞춰 별도로 작성하는 것을 권장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from .features import RawSample, build_feature_frame, build_feature_row, DEMOGRAPHIC_FIELDS
from .fusion_features import build_fusion_feature_row, build_fusion_feature_frame, FUSION_FEATURE_COLUMNS
from .fusion_model import FusionModel, FusionTrainingReport, MIN_FUSION_TRAINING_SAMPLES, _build_fusion_pipeline, IMPROVEMENT_THRESHOLD
from .legacy_scoring import METRIC_NAMES, compute_legacy_risk
from .model import RiskModel
from .train import _load_population_stats, _load_personal_stats, _legacy_scores


def _row_to_fusion_input(row: pd.Series, population_stats: dict, personal_stats_by_user: dict, v3_model: RiskModel | None):
    """라벨 CSV 한 줄 -> (v3_feature_row, ai_pred_proba, ai_risk_score, ai_tier, quantum_result)."""
    metrics = {name: (None if pd.isna(row.get(name)) else float(row[name])) for name in METRIC_NAMES}
    demographics = {f: (None if pd.isna(row.get(f)) else float(row[f])) for f in DEMOGRAPHIC_FIELDS}
    n_personal = int(row.get("n_personal", 0))

    sample = RawSample(metrics=metrics, demographics=demographics, n_personal=n_personal)
    uid = row.get("user_id")
    pers = personal_stats_by_user.get(uid, {})
    v3_row = build_feature_row(sample, population_stats, pers)

    if v3_model is not None:
        X = pd.DataFrame([v3_row])[list(v3_row.keys())]
        ai_pred_proba = float(v3_model.predict_risk_score(X)[0]) / 100.0
        ai_risk_score = ai_pred_proba * 100.0
        ai_tier = None  # v3 자체 tier 판정은 여기서 다루지 않음 (Fusion 학습엔 불필요)
    else:
        # v3 모델이 아직 없으면 legacy로 대체 (ai_pred_proba는 legacy risk_score/100)
        legacy_result = compute_legacy_risk(metrics, population_stats, pers)
        ai_risk_score = legacy_result["risk_score"]
        ai_pred_proba = ai_risk_score / 100.0
        ai_tier = legacy_result["tier"]

    boosted_str = row.get("qaoa_boosted_metrics", "")
    boosted_metrics = boosted_str.split("|") if isinstance(boosted_str, str) and boosted_str else []
    quantum_result = None
    if not pd.isna(row.get("qaoa_combined_z")):
        quantum_result = {
            "boosted_metrics": boosted_metrics,
            "combined_z": float(row.get("qaoa_combined_z")),
            "top_bitstring_probability": float(row.get("qaoa_top_probability", np.nan)),
            "qaoa_matched_true_optimum": bool(row.get("qaoa_matched_true_optimum", 0)),
            "risk_score": float(row.get("ai_risk_score", np.nan)),  # 비교용 — 실제 QAOA risk_score 컬럼 없으면 근사
            "tier": row.get("ai_tier"),
        }

    return v3_row, ai_pred_proba, ai_risk_score, ai_tier, quantum_result


def train_and_validate_fusion(
    df: pd.DataFrame,
    population_stats: dict,
    personal_stats_by_user: dict,
    v3_model: RiskModel | None,
):
    if len(df) < MIN_FUSION_TRAINING_SAMPLES:
        print(
            f"[경고] 라벨 샘플이 {len(df)}건으로 권장 최소치({MIN_FUSION_TRAINING_SAMPLES}건) 미만입니다. "
            "지금 학습 결과는 참고용이며, 실서비스 반영 전 반드시 데이터를 더 모아 재검증하세요."
        )

    fusion_rows = []
    for _, row in df.iterrows():
        v3_row, ai_pred_proba, ai_risk_score, ai_tier, quantum_result = _row_to_fusion_input(
            row, population_stats, personal_stats_by_user, v3_model
        )
        fusion_rows.append(
            build_fusion_feature_row(ai_pred_proba, ai_risk_score, ai_tier, quantum_result, v3_row)
        )

    X = build_fusion_feature_frame(fusion_rows)
    y = df["label"].to_numpy()

    n_splits = max(2, min(5, int(np.bincount(y.astype(int)).min()))) if len(np.unique(y)) == 2 else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_proba = cross_val_predict(_build_fusion_pipeline(), X, y, cv=skf, method="predict_proba")[:, 1]
    fold_aucs = [roc_auc_score(y[te], oof_proba[te]) for _, te in skf.split(X, y)]
    cv_auc_mean = float(np.mean(fold_aucs))
    cv_auc_std = float(np.std(fold_aucs))

    # legacy 비교 (train.py의 기존 유틸 재사용)
    legacy_proba = _legacy_scores(df, population_stats, personal_stats_by_user) / 100.0
    legacy_auc = float(roc_auc_score(y, legacy_proba)) if len(np.unique(y)) == 2 else None

    # v3 단독 비교 (ai_pred_proba 컬럼 자체를 v3의 예측치로 사용)
    v3_alone_auc = float(roc_auc_score(y, X["ai_pred_proba"])) if len(np.unique(y)) == 2 else None

    model = FusionModel()
    model.fit(X, y, calibrate=True)
    model.is_validated_better_than_legacy = bool(
        legacy_auc is not None and (cv_auc_mean - legacy_auc) > IMPROVEMENT_THRESHOLD
    )
    model.is_validated_better_than_v3 = bool(
        v3_alone_auc is not None and (cv_auc_mean - v3_alone_auc) > IMPROVEMENT_THRESHOLD
    )

    fitted_clf = model.pipeline.named_steps["clf"]
    importances = dict(zip(FUSION_FEATURE_COLUMNS, np.abs(fitted_clf.coef_[0])))

    report = FusionTrainingReport(
        n_samples=len(df), cv_auc_mean=cv_auc_mean, cv_auc_std=cv_auc_std,
        legacy_auc=legacy_auc, v3_alone_auc=v3_alone_auc, feature_importances=importances,
    )
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion(v4) 모델 훈련 및 legacy/v3 비교")
    parser.add_argument("--data", required=True)
    parser.add_argument("--population-stats", required=True)
    parser.add_argument("--personal-stats", required=True)
    parser.add_argument("--v3-model-dir", default=None, help="이미 학습된 v3 모델 폴더 (없으면 legacy로 대체)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    population_stats = _load_population_stats(args.population_stats)
    personal_stats_by_user = _load_personal_stats(args.personal_stats)

    v3_model = None
    if args.v3_model_dir:
        v3_model = RiskModel.load(args.v3_model_dir)
        if not v3_model.is_reliable():
            print("[알림] 지정된 v3 모델이 is_reliable()==False 입니다. ai_pred_proba는 legacy 기준으로 대체됩니다.")
            v3_model = None

    model, report = train_and_validate_fusion(df, population_stats, personal_stats_by_user, v3_model)

    print(report.summary())
    print("\n상위 피처 중요도 (|계수| 기준):")
    for name, importance in sorted(report.feature_importances.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name}: {importance:.3f}")

    model.save(args.out_dir)
    Path(args.out_dir, "training_report.json").write_text(
        json.dumps(report.__dict__, ensure_ascii=False, indent=2)
    )
    print(f"\n모델 저장 완료: {args.out_dir}")

    if model.is_validated_better_than_legacy and model.is_validated_better_than_v3:
        print(
            "\n" + "=" * 60 +
            "\n[승인 대기 — PENDING APPROVAL]" +
            "\nFusion이 legacy와 v3 단독 모두보다 유의미하게 우수합니다. 다만 통계적 조건을" +
            "\n만족해도 사람이 명시적으로 승인하기 전까지는 자동으로 반영되지 않습니다." +
            "\n" +
            "\n승인하려면:" +
            f"\n  python -m risk_model.approve_model --model-dir {args.out_dir} --approver \"<이름>\"" +
            "\n" + "=" * 60
        )
    else:
        print(
            "\n[알림] Fusion이 legacy 또는 v3 단독 대비 유의미한 개선을 보이지 않았습니다. "
            "승격 대상이 아닙니다 — v3 단독을 계속 사용하는 것을 권장합니다."
        )


if __name__ == "__main__":
    main()
