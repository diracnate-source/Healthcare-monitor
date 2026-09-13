# -*- coding: utf-8 -*-
"""
dual_path.py

검사 1회마다 AI 경로(v1/v2/v3 챔피언-챌린저)와 양자 경로(QAOA 가중치
최적화)를 동시에 계산하고, 둘 다 로그에 남기는 통합 모듈.

이게 왜 필요한가
------------------
지금까지 여러 턴에 걸쳐 만들어 온 두 갈래 — compute_risk_score_v3()가
산출하는 AI 경로(항상 화면에 노출되는 '공식' 결과)와,
quantum_optimizer.compute_quantum_optimized_score()가 산출하는 양자
경로(참고용 실험 결과) — 를 한 곳에서 같이 계산해 세션마다 나란히
기록한다. 지금은 라벨 데이터가 없어 "어느 쪽이 더 정확한가"를 판단할
수 없지만, PoC 운영을 통해 세션이 쌓이고 나중에 임상 라벨이 확보되면,
이 로그를 그대로 돌아봐서 "AI 경로와 양자 경로 중 어느 쪽이 실제
결과를 더 잘 예측했는가"를 사후적으로 검증할 수 있다.

원칙 (지금까지 일관되게 지켜온 것과 동일)
------------------------------------------
- 화면에 노출되는 '공식' 판정은 항상 AI 경로(compute_risk_score_v3)의
  결과다. 양자 경로는 이 함수를 통해 계산되고 로그에 남더라도, 그
  자체로 최종 판정에 관여하지 않는다.
- 양자 경로는 데이터기반 모드(z_plus_by_metric이 채워진 경우)에서만
  계산 가능하다. 부트스트랩 모드에서는 z-score 자체가 없어 QAOA
  입력을 만들 수 없으므로 quantum_result가 None으로 남는다 — 이것도
  로그에 정직하게 기록된다(가짜 값을 채우지 않는다).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from .app_py_adapter import compute_risk_score_v3, DEFAULT_MODEL_DIR, DEFAULT_POPULATION_STORE_PATH, DEFAULT_BASELINE_STORE_PATH

DEFAULT_DUAL_PATH_LOG_PATH = "dual_path_log.jsonl"

_log_lock = threading.Lock()

# QAOA 기본 설정 — quantum_optimizer.py의 값과 일치시킨다. 여기서 한
# 곳에 모아둬서, 나중에 실험 설계상 k_target/penalty를 바꾸게 되면
# 이 파일만 고치면 되게 한다.
QUANTUM_K_TARGET = 3
QUANTUM_PENALTY = 1.5


def compute_dual_path_result(
    metrics: dict,
    reaction_ms: Optional[float] = None,
    user_id: Optional[str] = None,
    demographics: Optional[dict] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    population_store_path: str = DEFAULT_POPULATION_STORE_PATH,
    baseline_store_path: str = DEFAULT_BASELINE_STORE_PATH,
    log_path: str = DEFAULT_DUAL_PATH_LOG_PATH,
    log_result: bool = True,
):
    """AI 경로와 양자 경로를 함께 계산하고, 기본적으로 로그에 남긴다.

    반환값: (risk_score, raw_scores, detail, quantum_result)
      - risk_score, raw_scores, detail: 기존 compute_risk_score_v3()와
        완전히 동일 — 화면에 표시되는 '공식' 결과. 이 함수를 쓰기 위해
        app.py의 기존 호출부·표시 로직을 바꿀 필요가 없다.
      - quantum_result: dict 또는 None. None이면 이번 세션은 부트스트랩
        모드라 양자 경로를 계산할 수 없었다는 뜻.
    """
    risk_score, raw_scores, detail = compute_risk_score_v3(
        metrics,
        reaction_ms=reaction_ms,
        user_id=user_id,
        demographics=demographics,
        model_dir=model_dir,
        population_store_path=population_store_path,
        baseline_store_path=baseline_store_path,
    )

    quantum_result = None
    z_plus_by_metric = detail.get("z_plus_by_metric")

    if z_plus_by_metric:
        from .quantum_optimizer import compute_quantum_optimized_score

        q = compute_quantum_optimized_score(
            z_plus_by_metric, k_target=QUANTUM_K_TARGET, penalty=QUANTUM_PENALTY
        )
        quantum_result = {
            "risk_score": q.risk_score,
            "tier": q.tier,
            "combined_z": q.combined_z,
            "boosted_metrics": q.boosted_metrics,
            "effective_weights": q.effective_weights,
            "qaoa_matched_true_optimum": q.qaoa_result.qaoa_matched_true_optimum,
            "k_target": QUANTUM_K_TARGET,
            "penalty": QUANTUM_PENALTY,
        }

    if log_result:
        _log_dual_path_result(
            user_id=user_id,
            ai_mode=detail.get("scored_by"),
            ai_risk_score=risk_score,
            ai_tier=detail.get("tier"),
            quantum_result=quantum_result,
            log_path=log_path,
        )

    return risk_score, raw_scores, detail, quantum_result


def _log_dual_path_result(
    user_id: Optional[str],
    ai_mode: Optional[str],
    ai_risk_score: float,
    ai_tier: Optional[str],
    quantum_result: Optional[dict],
    log_path: str,
) -> None:
    """세션 1건을 JSONL 한 줄로 append한다. 실패해도 위험도 계산
    파이프라인 자체에는 영향이 없도록 예외를 삼킨다 (감사용 부가 기능이
    핵심 기능을 막아서는 안 된다는 원칙 — stratified_stats.py와 동일)."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "ai_mode": ai_mode,
        "ai_risk_score": ai_risk_score,
        "ai_tier": ai_tier,
        "quantum_risk_score": quantum_result["risk_score"] if quantum_result else None,
        "quantum_tier": quantum_result["tier"] if quantum_result else None,
        "quantum_boosted_metrics": quantum_result["boosted_metrics"] if quantum_result else None,
        # 두 경로가 같은 3단계 판정에 도달했는지 — 라벨 없이도 지금 당장
        # 확인 가능한 '경로 간 일치율' 지표. 일치율 자체가 예측력을
        # 뜻하진 않지만, 두 경로가 얼마나 자주/어떤 조건에서 갈리는지는
        # 지금부터 관찰해 둘 가치가 있는 기초 통계다.
        "tiers_agree": (quantum_result["tier"] == ai_tier) if quantum_result else None,
    }

    try:
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def summarize_dual_path_log(log_path: str = DEFAULT_DUAL_PATH_LOG_PATH) -> dict:
    """지금까지 쌓인 로그의 기초 요약 — 라벨이 없는 지금 단계에서도
    확인 가능한 것만 담는다 (정확도 비교는 라벨이 있어야 가능하므로
    포함하지 않는다).
    """
    path = Path(log_path)
    if not path.exists():
        return {"n_sessions": 0}

    n_total = 0
    n_with_quantum = 0
    n_agree = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            n_total += 1
            if entry.get("quantum_risk_score") is not None:
                n_with_quantum += 1
                if entry.get("tiers_agree"):
                    n_agree += 1

    return {
        "n_sessions": n_total,
        "n_with_quantum_comparison": n_with_quantum,
        "n_tier_agreement": n_agree,
        "tier_agreement_rate": (n_agree / n_with_quantum) if n_with_quantum > 0 else None,
    }
