# -*- coding: utf-8 -*-
"""
cognitive_tasks_component  (V2 인지과제 1·2번)

브라우저 안에서 두 가지 반응 과제를 연속으로 실행하고, 시행별 기록을 한 번에 돌려준다.
  1) 단순 반응 (기본 10회)
  2) 선택 반응 Go/No-Go (기본 24회, Go 17 : No-Go 7)
모든 시간은 브라우저의 performance.now()로 재므로 서버 왕복 시간이 섞이지 않는다.

반환값(완료 시 1회):
  {
    "status": "ok",
    "reaction_ms": <단순 반응 중앙값 ms 또는 None>,   # 기존 점수 계산과의 호환용
    "simple":  {"trials": [...], "summary": {...}},
    "gonogo":  {"trials": [...], "summary": {...}},
    "device":  {...}
  }
연구용 시제품이며 진단 목적이 아니다. 문항·기준은 임상 검증 전이다.
"""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_component_func = components.declare_component(
    "cognitive_tasks",
    path=_COMPONENT_DIR,
)


def cognitive_tasks_component(
    simple_n: int = 10,
    gonogo_n: int = 24,
    time_scale: float = 1.0,
    key: str = None,
):
    """time_scale: 대기·제시 시간을 곱하는 배율(테스트용, 실사용은 1.0)."""
    return _component_func(
        simple_n=simple_n,
        gonogo_n=gonogo_n,
        time_scale=time_scale,
        key=key,
        default=None,
    )
