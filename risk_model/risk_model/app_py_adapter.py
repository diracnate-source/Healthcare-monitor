# -*- coding: utf-8 -*-
"""
app_py_adapter.py

원본 app.py의 데이터/함수 시그니처를 risk_model 패키지(legacy_scoring,
features, model)와 연결하는 어댑터.

app.py를 크게 뜯어고치지 않고, 다음 한 줄만 교체하는 것을 목표로 설계했다.

    # 기존 (app.py 2316행)
    risk_score, raw_scores, detail = compute_risk_score(
        screening["metrics"], reaction_ms=reaction_ms, user_id=(user_id or None)
    )

    # 교체 후
    from risk_model.app_py_adapter import compute_risk_score_v3
    risk_score, raw_scores, detail = compute_risk_score_v3(
        screening["metrics"], reaction_ms=reaction_ms, user_id=(user_id or None)
    )

반환 타입(risk_score: float, raw_scores: dict, detail: dict)은 원본과
동일하게 유지해서, 화면 렌더링 코드(2338~2419행)를 그대로 재사용할 수
있게 했다. 다만 detail["mode"]에 새 값 "learned_model"이 추가될 수 있으므로,
화면 쪽에 분기 하나를 추가해야 한다 — 이 파일 맨 아래 주석에 그 패치를
같이 적어두었다.

전제: population_store.json / baseline_store.json 은 원본 app.py가 쓰던
스키마 그대로다.
  population_store.json: {"expression_change": {"n": int, "mean": float, "m2": float}, ...}
  baseline_store.json:   {"<user_id>": {"n_sessions": int, "metric_means": {...}, ...}, ...}
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

from .features import DEMOGRAPHIC_FIELDS, RawSample, build_feature_frame
from .legacy_scoring import (
    LEGACY_WEIGHTS,
    METRIC_LABELS,
    METRIC_NAMES,
    MIN_POPULATION_N,
    PersonalStats,
    WelfordAccumulator,
    _legacy_reaction_raw,
    _tier_from_score,
    _tier_from_z,
    compute_legacy_risk,
)
from .model import RiskModel

# app.py와 동일한 기본 파일명 (배포 환경에서 경로가 다르면 인자로 덮어쓴다)
DEFAULT_POPULATION_STORE_PATH = "population_store.json"
DEFAULT_BASELINE_STORE_PATH = "baseline_store.json"
DEFAULT_MODEL_DIR = "./trained_model"

_model_cache_lock = threading.Lock()
_cached_model: Optional[RiskModel] = None
_cached_model_dir: Optional[str] = None


# ------------------------------------------------------------
# 1) app.py 저장소(JSON) -> risk_model 통계 객체 변환
# ------------------------------------------------------------

def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def convert_app_population_store(
    path: str = DEFAULT_POPULATION_STORE_PATH,
) -> Dict[str, WelfordAccumulator]:
    """population_store.json -> {지표명: WelfordAccumulator}.

    아직 한 번도 기록되지 않은 지표는 n=0인 빈 accumulator로 채운다.
    (n=0 이면 std=None, ready=False 가 되어 자연스럽게 부트스트랩
    모드로 폴백하는 원본 동작과 동일하다.)
    """
    store = _load_json(path)
    stats: Dict[str, WelfordAccumulator] = {}

    for name in METRIC_NAMES:
        entry = store.get(name)
        acc = WelfordAccumulator()
        if entry is not None:
            acc.n = int(entry.get("n", 0))
            acc.mean = float(entry.get("mean", 0.0))
            acc.m2 = float(entry.get("m2", 0.0))
        stats[name] = acc

    return stats


def convert_app_personal_store(
    user_id: Optional[str],
    population_stats: Dict[str, WelfordAccumulator],
    path: str = DEFAULT_BASELINE_STORE_PATH,
) -> Dict[str, PersonalStats]:
    """baseline_store.json의 특정 user_id 항목 -> {지표명: PersonalStats}.

    원본 app.py는 지표별로 세션 수를 따로 세지 않고 n_sessions 하나를
    모든 지표가 공유한다(get_personal_stats). 그 동작을 그대로 재현한다:
    - n_personal = entry["n_sessions"] (모든 지표 공통)
    - mean = entry["metric_means"][지표] 가 있으면 그 값,
      없으면(그 지표가 한 번도 유효하게 기록된 적 없으면) 모집단 평균으로
      대체 — 이렇게 하면 EB 결합 시 mu_ref가 그냥 모집단 평균으로
      수렴해 원본과 동일하게 동작한다.
    """
    result: Dict[str, PersonalStats] = {}
    if not user_id:
        for name in METRIC_NAMES:
            result[name] = PersonalStats(n_personal=0, mean=0.0)
        return result

    store = _load_json(path)
    entry = store.get(user_id)
    n_sessions = int(entry.get("n_sessions", 0)) if entry else 0
    metric_means = entry.get("metric_means", {}) if entry else {}

    for name in METRIC_NAMES:
        pop = population_stats.get(name)
        fallback_mean = pop.mean if pop is not None else 0.0
        mean = float(metric_means.get(name, fallback_mean))
        result[name] = PersonalStats(n_personal=n_sessions, mean=mean)

    return result


# ------------------------------------------------------------
# 2) app.py 지표 dict -> RawSample
# ------------------------------------------------------------

def build_raw_sample(
    metrics: dict,
    reaction_ms: Optional[float],
    n_personal: int,
    demographics: Optional[dict] = None,
    sample_id: Optional[str] = None,
) -> RawSample:
    """generate_landmark_visualization_and_metrics()의 반환 dict +
    reaction_ms + (선택) 인구통계를 RawSample로 변환한다.

    주의: 지금 app.py 화면에는 age/sex/education_years를 입력받는 UI가
    없다. demographics를 안 넘기면 전부 NaN으로 채워지고, 모델은 그
    피처들을 결측으로 처리한다 — 즉 지금 당장은 인구통계 없이도 동작은
    하지만, 실제로 이 피처가 위험도 산출에 기여하게 하려면 report 단계
    (2301행 부근, user_id 입력받는 곳)에 나이 등을 입력받는 UI를 추가해야
    한다.
    """
    demographics = demographics or {}
    metric_values = {name: metrics.get(name) for name in METRIC_NAMES if name != "reaction_ms"}
    metric_values["reaction_ms"] = reaction_ms

    return RawSample(
        metrics=metric_values,
        demographics={field: demographics.get(field) for field in DEMOGRAPHIC_FIELDS},
        n_personal=n_personal,
        sample_id=sample_id,
    )


# ------------------------------------------------------------
# 3) 학습된 모델 로드 (캐시)
# ------------------------------------------------------------

def _get_cached_model(model_dir: str) -> Optional[RiskModel]:
    """디스크에서 RiskModel을 로드하되, 같은 경로에 대해서는 프로세스
    수명 동안 한 번만 로드한다. 모델 파일이 아직 없으면(콜드스타트)
    조용히 None을 반환한다 — 이 경우 이후 로직이 자동으로 legacy로
    폴백한다.

    Streamlit 앱에서는 이 함수 대신 @st.cache_resource로 감싼 래퍼를
    써도 된다. 여기서는 risk_model 패키지가 streamlit에 의존하지 않도록
    threading.Lock 기반의 단순 캐시를 직접 구현했다.
    """
    global _cached_model, _cached_model_dir

    with _model_cache_lock:
        if _cached_model is not None and _cached_model_dir == model_dir:
            return _cached_model

        model_path = Path(model_dir)
        if not (model_path / "pipeline.joblib").exists():
            return None

        try:
            model = RiskModel.load(model_dir)
        except Exception:
            # 모델 파일이 손상되었거나 스키마가 안 맞는 경우에도 앱이
            # 죽지 않고 legacy로 계속 동작해야 한다.
            return None

        _cached_model = model
        _cached_model_dir = model_dir
        return model


# ------------------------------------------------------------
# 4) 화면 표시용 raw_scores 재구성 (원본과 동일한 형식 유지)
# ------------------------------------------------------------

def _build_display_raw_scores(
    metrics: dict,
    reaction_ms: Optional[float],
    mode: str,
    legacy_detail: Optional[dict],
) -> Dict[str, Optional[float]]:
    """expander에 보여줄 지표별 참고값을, 원본 compute_risk_score()가
    만들던 형식(한글 라벨 -> 값) 그대로 재구성한다.

    - bootstrap 모드: 원시값 x 고정계수
    - data_driven 모드: legacy_detail["detail"][metric]["z_plus"]
    - learned_model 모드: 학습모델은 지표별 z+ 개념이 없으므로, 원시
      측정값 자체를 보여준다 (참고용이라는 caption은 app.py 쪽에서
      모드에 맞게 문구를 하나 더 추가해줘야 한다 — 아래 패치 참고).
    """
    raw_scores: Dict[str, Optional[float]] = {}

    if mode == "bootstrap":
        for name, weight in LEGACY_WEIGHTS.items():
            x = metrics.get(name)
            raw_scores[METRIC_LABELS[name]] = round(x * weight, 2) if x is not None else None
        raw_scores[METRIC_LABELS["reaction_ms"]] = (
            round(_legacy_reaction_raw(reaction_ms), 2) if reaction_ms is not None else None
        )
        return raw_scores

    if mode == "data_driven" and legacy_detail is not None:
        detail_map = legacy_detail.get("detail", {})
        for name in METRIC_NAMES:
            entry = detail_map.get(name)
            raw_scores[METRIC_LABELS[name]] = round(entry["z_plus"], 3) if entry else None
        return raw_scores

    # learned_model 등: 원시값 그대로 표시
    for name in METRIC_NAMES:
        x = metrics.get(name) if name != "reaction_ms" else reaction_ms
        raw_scores[METRIC_LABELS[name]] = round(x, 2) if x is not None else None
    return raw_scores


# ------------------------------------------------------------
# 5) app.py의 compute_risk_score()를 대체하는 진입점
# ------------------------------------------------------------

def compute_risk_score_v3(
    metrics: dict,
    reaction_ms: Optional[float] = None,
    user_id: Optional[str] = None,
    demographics: Optional[dict] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    population_store_path: str = DEFAULT_POPULATION_STORE_PATH,
    baseline_store_path: str = DEFAULT_BASELINE_STORE_PATH,
):
    """app.py 원본 compute_risk_score()와 동일한 시그니처/반환 형태.

    내부적으로는:
      1) population_store.json / baseline_store.json 을 읽어 통계 객체로 변환
      2) 학습된 모델이 있고 신뢰 가능(is_reliable())하면 그 결과를 최종
         risk_score로 사용
      3) 모델이 없거나 신뢰 불가하면 legacy(v2) 결과를 그대로 사용
         (원본 app.py와 100% 동일한 값이 나온다)
      4) 어느 경로를 썼든, 화면 표시용 raw_scores/detail은 항상
         legacy 계산도 함께 수행해 만든다 — 그래야 "지금 legacy라면
         몇 점이 나왔을지"를 detail["legacy_risk_score"]로 비교해볼 수
         있다.
    """
    population_stats = convert_app_population_store(population_store_path)
    personal_stats = convert_app_personal_store(user_id, population_stats, baseline_store_path)

    # n_personal은 원본과 동일하게 baseline_store의 n_sessions를 그대로 쓴다
    n_personal = next(iter(personal_stats.values())).n_personal if personal_stats else 0

    sample = build_raw_sample(metrics, reaction_ms, n_personal, demographics)

    # 표시/비교용: legacy 결과는 항상 계산해둔다
    raw_values = {name: sample.metrics.get(name) for name in METRIC_NAMES}
    legacy_result = compute_legacy_risk(raw_values, population_stats, personal_stats)

    model = _get_cached_model(model_dir)

    if model is not None and model.is_reliable():
        feature_row = build_feature_frame(
            [sample], population_stats, {user_id: personal_stats} if user_id else {}, [user_id]
        )
        risk_score = float(model.predict_risk_score(feature_row)[0])
        tier = _tier_from_score(risk_score)
        mode = "learned_model"
        combined_z = None
    else:
        risk_score = legacy_result["risk_score"]
        tier = legacy_result["tier"]
        mode = legacy_result["mode"]  # "bootstrap" 또는 "data_driven"
        combined_z = legacy_result.get("combined_z")

    raw_scores = _build_display_raw_scores(
        raw_values, reaction_ms, legacy_result["mode"] if mode == "learned_model" else mode, legacy_result
    )

    population_n = {name: population_stats[name].n for name in METRIC_NAMES}

    detail = {
        # app.py 화면 로직 호환: 기존 값("bootstrap_legacy"/"data_driven")은
        # 그대로 두고, 학습모델 사용 시에만 새 값을 추가한다.
        "mode": {"bootstrap": "bootstrap_legacy", "data_driven": "data_driven"}.get(mode, mode),
        "combined_z": combined_z,
        "tier": tier,
        "n_personal": n_personal,
        "population_n": population_n,
        # --- 아래는 원본에 없던 신규 필드 (app.py 쪽에서 선택적으로 사용) ---
        "legacy_risk_score": legacy_result["risk_score"],
        "legacy_tier": legacy_result["tier"],
        "scored_by": mode,  # "bootstrap" / "data_driven" / "learned_model"
    }

    return risk_score, raw_scores, detail


# ------------------------------------------------------------
# app.py 패치 가이드 (실제 파일은 건드리지 않음 — 참고용 diff)
# ------------------------------------------------------------
#
# 1) import 추가 (파일 상단, mediapipe import 근처)
#
#     from risk_model.app_py_adapter import compute_risk_score_v3
#
# 2) 2316행 교체
#
#     - risk_score, raw_scores, detail = compute_risk_score(
#     + risk_score, raw_scores, detail = compute_risk_score_v3(
#           screening["metrics"],
#           reaction_ms=reaction_ms,
#           user_id=(user_id or None)
#       )
#
# 3) 2379행 expander 안내문 분기에 학습모델 케이스 추가
#
#     if detail["mode"] == "bootstrap_legacy":
#         st.caption("...")
#     elif detail.get("scored_by") == "learned_model":
#         st.caption(
#             "🤖 학습된 위험도 모델(학습 기반)로 계산된 결과입니다. "
#             f"참고: 같은 조건에서 기존 통계 기반 방식은 "
#             f"{detail['legacy_risk_score']:.1f}점이었습니다."
#         )
#     else:
#         st.caption("...")  # 기존 data_driven 문구 그대로
#
# 이 세 군데 외에는 app.py를 수정할 필요가 없다 — 저장 로직
# (update_population_stats, save_baseline)은 원본 그대로 두면 된다.
# 학습 모델은 population_store.json/baseline_store.json을 직접 쓰지 않고
# 읽기만 하므로, 기존 저장 파이프라인과 충돌하지 않는다.
