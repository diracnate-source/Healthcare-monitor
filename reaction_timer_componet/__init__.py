# -*- coding: utf-8 -*-
"""
reaction_timer_component

브라우저(클라이언트) 안에서 자극 표시~클릭까지의 시간을 직접 재는
Streamlit 커스텀 컴포넌트. 서버가 시간을 재던 기존 방식과 달리,
네트워크 왕복 시간이 측정값에 섞이지 않는다.

반환값은 다음 중 하나:
  None                                  # 아직 클릭 전
  {"status": "ok", "reaction_ms": 320.5}     # 정상 측정
  {"status": "early", "reaction_ms": None}   # 초록불 뜨기 전에 클릭(false start)
"""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_component_func = components.declare_component(
    "reaction_timer",
    path=_COMPONENT_DIR,
)


def reaction_timer_component(target_delay_ms: float, key: str = None):
    """target_delay_ms: '준비하세요' 화면을 얼마나 오래 보여준 뒤
    초록불(자극)을 띄울지, 밀리초 단위. 매 라운드(특히 false start 이후
    재시도)마다 다른 key를 넘겨야 컴포넌트가 완전히 새로 마운트되어
    내부 JS 상태(timerStarted 등)가 초기화된다.
    """
    return _component_func(target_delay_ms=target_delay_ms, key=key, default=None)
