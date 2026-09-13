# -*- coding: utf-8 -*-
"""
train.py

라벨(진단/MMSE 등)이 달린 세션 데이터로 RiskModel을 학습하고,
legacy v2(EB/Z-score) 방식과 AUC/민감도/특이도를 비교한다.

입력 CSV 스키마 (예시)
----------------------
sample_id, user_id,
expression_change, micro_movement, facial_asymmetry, blink_rate,
gaze_variability, reaction_ms,
age, sex, education_years,
n_personal,
label   (0=정상, 1=위험군 — 진단 또는 MMSE 컷오프 기준으로 사전 변환)

population_stats.json
----------------------
학습 시점까지 누적된 모집단 Welford 통계.
{"expression_change": {"n": 812, "mean": 4.1, "m2": ...}, ...}

personal_stats.json
--------------------
{"user_id": {"expression_change": {"n_personal": 5, "mean": 4.3}, ...}, ...}

사용법
------
python train.py \
    --data labeled_sessions.csv \
    --population-stats population_stats.json \
    --personal-stats personal_stats.json \
    --out-dir ./trained_model

주의: 이 스크립트는 데이터 파이프라인의 '뼈대'이다. 실제 라벨 정의
(어떤 진단기준을 이진 라벨로 쓸지), 클래스 불균형 처리, 다기관 데이터의
분포차 보정 등은 임상/통계 담당자와 함께 별도로 설계해야 한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix

from .features import FEATURE_COLUMNS, DEMOGRAPHIC_FIELDS, RawSample, build_feature_frame
from .legacy_scoring import (
    METRIC_NAMES,
    PersonalStats,
    WelfordAccumulator,
    compute_legacy_risk,
)
from .model import RiskModel, TrainingReport, MIN_TRAINING_SAMPLES, _build_pipeline


def _load_population_stats(path: str) -> dict:
    raw = json.loads(Path(path).read_text())
    stats = {}
    for name, d in raw.items():
        acc = WelfordAccumulator()
        acc.n = d["n"]
        acc.mean = d["mean"]
        acc.m2 = d["m2"]
        stats[name] = acc
    return stats


def _load_personal_stats(path: str) -> dict:
    raw = json.loads(Path(path).read_text())
    out = {}
    for user_id, metrics in raw.items():
        out[user_id] = {}
        for name, d in metrics.items():
            ps = PersonalStats()
            ps.n_personal = d["n_personal"]
            ps.mean = d["mean"]
            out[user_id][name] = ps
    return out


def _row_to_raw_sample(row: pd.Series) -> RawSample:
    metrics = {name: (None if pd.isna(row.get(name)) else float(row[name])) for name in METRIC_NAMES}
    demographics = {
        field: (None if pd.isna(row.get(field)) else float(row[field])) for field in DEMOGRAPHIC_FIELDS
    }
    return RawSample(
        metrics=metrics,
        demographics=demographics,
        n_personal=int(row.get("n_personal", 0)),
        label=None if pd.isna(row.get("label")) else float(row["label"]),
        sample_id=str(row.get("sample_id")),
    )


def _legacy_scores(df: pd.DataFrame, population_stats: dict, personal_stats_by_user: dict) -> np.ndarray:
    scores = []
    for _, row in df.iterrows():
        metrics = {name: (None if pd.isna(row.get(name)) else float(row[name])) for name in METRIC_NAMES}
        uid = row.get("user_id")
        pstats = personal_stats_by_user.get(uid, {})
        result = compute_legacy_risk(metrics, population_stats, pstats)
        scores.append(result["risk_score"])
    return np.array(scores)


def _sensitivity_specificity(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sensitivity, specificity


def train_and_validate(
    df: pd.DataFrame,
    population_stats: dict,
    personal_stats_by_user: dict,
) -> tuple[RiskModel, TrainingReport]:
    if len(df) < MIN_TRAINING_SAMPLES:
        print(
            f"[경고] 라벨 샘플이 {len(df)}건으로 권장 최소치({MIN_TRAINING_SAMPLES}건) 미만입니다. "
            "지금 학습 결과는 참고용이며, 실서비스 반영 전 반드시 데이터를 더 모아 재검증하세요."
        )

    samples = [_row_to_raw_sample(row) for _, row in df.iterrows()]
    user_ids = df["user_id"].tolist()
    X = build_feature_frame(samples, population_stats, personal_stats_by_user, user_ids)
    y = df["label"].to_numpy()

    # 5-fold 교차검증 (라벨 불균형을 고려해 StratifiedKFold)
    n_splits = min(5, int(np.bincount(y.astype(int)).min())) if len(np.unique(y)) == 2 else 5
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_proba = cross_val_predict(
        _build_pipeline(), X[FEATURE_COLUMNS], y, cv=skf, method="predict_proba"
    )[:, 1]

    fold_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        fold_aucs.append(roc_auc_score(y[test_idx], oof_proba[test_idx]))

    cv_auc_mean = float(np.mean(fold_aucs))
    cv_auc_std = float(np.std(fold_aucs))
    sensitivity, specificity = _sensitivity_specificity(y, oof_proba)

    legacy_proba = _legacy_scores(df, population_stats, personal_stats_by_user) / 100.0
    legacy_auc = float(roc_auc_score(y, legacy_proba)) if len(np.unique(y)) == 2 else None

    # 최종 모델은 전체 데이터로 재학습
    model = RiskModel()
    model.fit(X, y, calibrate=True)
    model.is_validated_better_than_legacy = bool(
        legacy_auc is not None and (cv_auc_mean - legacy_auc) > 0.02
    )

    fitted_clf = model.pipeline.named_steps["clf"]
    importances = dict(zip(FEATURE_COLUMNS, np.abs(fitted_clf.coef_[0])))

    report = TrainingReport(
        n_samples=len(df),
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        cv_sensitivity_mean=float(sensitivity),
        cv_specificity_mean=float(specificity),
        legacy_auc=legacy_auc,
        feature_importances=importances,
    )
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(description="위험도 학습 모델 훈련 및 legacy 비교")
    parser.add_argument("--data", required=True, help="라벨된 세션 CSV 경로")
    parser.add_argument("--population-stats", required=True, help="모집단 Welford 통계 JSON")
    parser.add_argument("--personal-stats", required=True, help="사용자별 개인 통계 JSON")
    parser.add_argument("--out-dir", required=True, help="학습된 모델 저장 경로")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    population_stats = _load_population_stats(args.population_stats)
    personal_stats_by_user = _load_personal_stats(args.personal_stats)

    model, report = train_and_validate(df, population_stats, personal_stats_by_user)

    print(report.summary())
    print("\n상위 피처 중요도 (|계수| 기준):")
    for name, importance in sorted(report.feature_importances.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name}: {importance:.3f}")

    model.save(args.out_dir)
    Path(args.out_dir, "training_report.json").write_text(
        json.dumps(report.__dict__, ensure_ascii=False, indent=2)
    )
    print(f"\n모델 저장 완료: {args.out_dir}")

    if not model.is_validated_better_than_legacy:
        print(
            "\n[알림] 이 모델은 아직 legacy v2 대비 유의미한 성능 우위가 확인되지 않았습니다. "
            "서비스에서는 계속 legacy 폴백을 사용하는 것을 권장합니다 (model.is_reliable() == False)."
        )
    else:
        print(
            "\n" + "=" * 60 +
            "\n[승인 대기 — PENDING APPROVAL]" +
            "\n이 모델은 legacy 대비 통계적 우위가 확인되었지만, 아직 아무도 승인하지" +
            "\n않았습니다 (is_approved=False). 통계적 조건을 만족해도 사람이 명시적으로" +
            "\n승인하기 전까지는 자동으로 프로덕션에 반영되지 않습니다 (의료 목적" +
            "\n시스템에서 알고리즘이 사전 승인 없이 스스로 바뀌는 것을 막기 위한 안전장치)." +
            "\n" +
            "\n승인하려면 위 training_report.json 요약을 검토한 뒤 아래 명령을 실행하세요:" +
            f"\n  python -m risk_model.approve_model --model-dir {args.out_dir} --approver \"<이름>\"" +
            "\n" + "=" * 60
        )


if __name__ == "__main__":
    main()
