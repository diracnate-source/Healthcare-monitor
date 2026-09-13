# -*- coding: utf-8 -*-
"""
stratified_stats.py

인구통계(연령대·성별 등)별로 모집단 통계를 별도 구획에 나눠 저장해,
나중에 "이 시스템이 특정 하위집단에서 유독 다르게 동작하지는 않는가"를
감사(audit)할 수 있게 하는 모듈.

왜 필요한가
------------
웹캠 기반 얼굴 분석은 피부톤·조명·연령·성별에 따라 성능이 다르게 나올 수
있다는 것이 컴퓨터 비전 분야에 이미 보고된 편향 문제다. 그런데 지금까지
population_store.json은 모든 사용자를 하나로 합친 평균·분산만 저장해서,
특정 하위집단만 유독 지표 분포가 다르게 나오는지조차 나중에 확인할
방법이 없었다.

지금 이 모듈이 하는 일과 하지 않는 일
--------------------------------------
- 한다: 인구통계가 있는 세션에 한해, 전체(overall) 통계와는 별도로
  하위집단(stratum)별 Welford 통계를 추가로 누적해 저장한다.
- 하지 않는다: 지금 당장 위험도 점수 계산 방식을 하위집단별로
  바꾸지 않는다. app.py에 인구통계 입력 UI가 아직 없어 대부분의 세션은
  '미상(unknown)' 구획에 쌓일 것이고, 그 자체가 이미 유의미한 정보다
  (얼마나 많은 세션에 인구통계가 비어있는지). 표본이 하위집단별로
  충분히 쌓이기 전에 계산 방식부터 바꾸면, 오히려 근거 없는 세분화가
  된다. 지금은 '나중에 감사할 수 있도록 데이터를 계층화해서 쌓아두는
  것'까지가 목표다.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

DEFAULT_STRATIFIED_STORE_PATH = "population_store_stratified.json"

_lock = threading.Lock()

# 연령 구간 경계. 이 경계 자체도 실제 사용자 분포가 쌓이면 재검토해야
# 하는 잠정값이다.
AGE_BUCKETS = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 200)]


def _age_bucket_label(age: Optional[float]) -> str:
    if age is None:
        return "age_unknown"
    for lo, hi in AGE_BUCKETS:
        if lo <= age < hi:
            return f"age_{lo}-{hi}"
    return "age_unknown"


def _sex_label(sex: Optional[float]) -> str:
    # app.py의 인구통계 입력 스키마와 맞춰야 한다 (0/1 인코딩 가정).
    # 아직 입력 UI가 없어 대부분 None(미상)으로 들어올 것이다.
    if sex is None:
        return "sex_unknown"
    return "sex_F" if sex == 0 else "sex_M"


def stratum_key(demographics: Optional[Dict[str, Optional[float]]]) -> str:
    """인구통계 dict -> 계층 키 문자열 (예: 'age_60-70|sex_F').
    demographics가 아예 없거나 필드가 비어 있으면 'age_unknown|sex_unknown'.
    """
    demographics = demographics or {}
    age_label = _age_bucket_label(demographics.get("age"))
    sex_label = _sex_label(demographics.get("sex"))
    return f"{age_label}|{sex_label}"


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_stratified_stats(
    metrics: Dict[str, Optional[float]],
    reaction_ms: Optional[float],
    demographics: Optional[Dict[str, Optional[float]]],
    path: str = DEFAULT_STRATIFIED_STORE_PATH,
) -> None:
    """이번 세션의 지표값들을, 해당 인구통계 구획의 Welford 통계에
    추가로 누적한다. 기존 population_store.json(전체 통합 통계)과는
    별도 파일이라, 기존 파이프라인(위험도 계산)에는 전혀 영향을 주지
    않는다 — 순수하게 감사용 부가 저장소다.
    """
    key = stratum_key(demographics)
    values = dict(metrics)
    values["reaction_ms"] = reaction_ms

    with _lock:
        store = _load(path)
        stratum_store = store.get(key, {})

        for metric_name, x in values.items():
            if x is None:
                continue
            entry = stratum_store.get(metric_name, {"n": 0, "mean": 0.0, "m2": 0.0})
            n = entry["n"] + 1
            delta = x - entry["mean"]
            mean = entry["mean"] + delta / n
            delta2 = x - mean
            m2 = entry["m2"] + delta * delta2
            stratum_store[metric_name] = {"n": n, "mean": mean, "m2": m2}

        store[key] = stratum_store

        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)


def summarize_strata(path: str = DEFAULT_STRATIFIED_STORE_PATH) -> Dict[str, Dict[str, int]]:
    """감사용 요약: 구획별로 몇 건씩 쌓였는지 지표별 표본 수를 반환한다.
    (성능 차이를 논하기엔 아직 라벨이 없어 이르지만, '표본이 하위집단별로
    고르게 쌓이고 있는가' 자체는 지금도 확인할 수 있는 유용한 정보다.)
    """
    store = _load(path)
    summary = {}
    for stratum, metrics in store.items():
        summary[stratum] = {name: entry.get("n", 0) for name, entry in metrics.items()}
    return summary
