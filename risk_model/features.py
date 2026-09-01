# -*- coding: utf-8 -*-
"""
features.py

"6개 지표 + 인구통계(나이 등) + 개인/모집단 통계 -> 학습된 모델 -> risk_score"
구조에서, 학습 모델에 들어갈 피처 벡터를 만드는 부분.

설계 원칙
---------
1. 원시 지표(raw metric) 값 자체뿐 아니라, 그 값이 개인/모집단 기준에서
   얼마나 벗어나 있는지(z-score류 파생값)도 함께 넣는다.
   -> legacy_scoring.py 처럼 "사람이 정한 공식으로 미리 z-score를 계산해
      넣고 그 뒤엔 고정 가중합만 한다"가 아니라, 모델이 원시값과 편차값을
      동시에 보고 상호작용(예: '평소 대비 편차가 클 때 asymmetry가 특히
      더 위험하다')을 스스로 학습할 여지를 남긴다.
2. 결측(None)은 -1 등으로 임의 대체하지 않고, LightGBM/XGBoost류가 처리하는
   NaN 그대로 흘려보낸다. 로지스틱 회귀처럼 결측을 못 받는 모델을 쓸 경우엔
   train.py 단계에서 SimpleImputer로 별도 처리한다(무엇을 대치했는지 항상
   'was_missing' 플래그 피처를 같이 만들어 정보 손실을 줄인다).
3. 개인 통계가 아직 없는 사용자(n_personal=0, 콜드스타트)는 개인편차 피처가
   정의되지 않으므로 population 평균/표준편차만으로 만든 값 + n_personal
   자체를 피처로 넣어 모델이 '이 사람은 아직 개인 기준선이 없다'는 사실도
   학습할 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .legacy_scoring import METRIC_NAMES, PersonalStats, WelfordAccumulator

DEMOGRAPHIC_FIELDS = ("age", "sex", "education_years")


@dataclass
class RawSample:
    """모델 학습/추론에 필요한 한 세션의 원시 입력."""

    metrics: Dict[str, Optional[float]]           # 6개 지표 원시값
    demographics: Dict[str, Optional[float]]       # age, sex(0/1), education_years 등
    n_personal: int                                 # 이 사용자의 누적 세션 수 (콜드스타트 신호)
    label: Optional[float] = None                   # 학습 시에만 사용 (예: MMSE 점수 또는 진단 이진값)
    sample_id: Optional[str] = None


def build_feature_row(
    sample: RawSample,
    population_stats: Dict[str, WelfordAccumulator],
    personal_stats: Dict[str, PersonalStats],
) -> Dict[str, float]:
    """RawSample 하나 -> {피처명: 값} 딕셔너리.

    피처 구성 (지표당 4개 + 인구통계 + 메타 2개):
      {metric}_raw        원시 지표값
      {metric}_missing    결측 여부 (1/0)
      {metric}_pop_z      모집단 기준 z ( (x-pop_mean)/pop_std ), 부호 유지(단측 아님)
      {metric}_personal_z 개인 기준 z ( (x-personal_mean)/pop_std ), 개인데이터 없으면 NaN
      age, sex, education_years ...
      n_personal           누적 세션 수 (콜드스타트 정도를 모델이 직접 인지)
      population_ready_frac  전체 지표 중 모집단 통계가 '신뢰 가능(ready)'한 비율
    """
    row: Dict[str, float] = {}

    ready_count = 0
    for name in METRIC_NAMES:
        x = sample.metrics.get(name)
        pop = population_stats.get(name)
        pers = personal_stats.get(name)

        row[f"{name}_raw"] = np.nan if x is None else float(x)
        row[f"{name}_missing"] = 1.0 if x is None else 0.0

        if x is not None and pop is not None and pop.std is not None and pop.std > 1e-6:
            row[f"{name}_pop_z"] = (x - pop.mean) / pop.std
            if pop.ready:
                ready_count += 1
        else:
            row[f"{name}_pop_z"] = np.nan

        if (
            x is not None
            and pers is not None
            and pers.n_personal > 0
            and pop is not None
            and pop.std is not None
            and pop.std > 1e-6
        ):
            row[f"{name}_personal_z"] = (x - pers.mean) / pop.std
        else:
            row[f"{name}_personal_z"] = np.nan

    for field_name in DEMOGRAPHIC_FIELDS:
        val = sample.demographics.get(field_name)
        row[field_name] = np.nan if val is None else float(val)

    row["n_personal"] = float(sample.n_personal)
    row["population_ready_frac"] = ready_count / len(METRIC_NAMES)

    return row


def build_feature_frame(
    samples: list[RawSample],
    population_stats: Dict[str, WelfordAccumulator],
    personal_stats_by_user: Dict[str, Dict[str, PersonalStats]],
    user_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """여러 샘플을 한 번에 피처 데이터프레임으로 변환.

    personal_stats_by_user: {user_id: {metric_name: PersonalStats}}
    user_ids: samples와 같은 길이. 각 샘플이 어느 사용자의 것인지.
    """
    rows = []
    for i, sample in enumerate(samples):
        uid = user_ids[i] if user_ids is not None else None
        personal_stats = personal_stats_by_user.get(uid, {}) if uid is not None else {}
        rows.append(build_feature_row(sample, population_stats, personal_stats))
    frame = pd.DataFrame(rows)
    if any(s.sample_id for s in samples):
        frame.index = [s.sample_id for s in samples]
    return frame


FEATURE_COLUMNS = (
    [f"{m}_raw" for m in METRIC_NAMES]
    + [f"{m}_missing" for m in METRIC_NAMES]
    + [f"{m}_pop_z" for m in METRIC_NAMES]
    + [f"{m}_personal_z" for m in METRIC_NAMES]
    + list(DEMOGRAPHIC_FIELDS)
    + ["n_personal", "population_ready_frac"]
)
