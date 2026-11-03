# -*- coding: utf-8 -*-
"""
voice_timing_component

마이크 입력의 '소리 크기 곡선'만 브라우저 안에서 분석해, 말한 구간·쉰 구간의
시간 정보를 돌려주는 Streamlit 커스텀 컴포넌트.

- 소리 원본은 녹음·저장·전송하지 않는다 (MediaRecorder 미사용, AnalyserNode만 사용).
- 단어를 인식하지 않는다. 말 시작까지의 시간, 말한 구간 수·길이, 쉰 구간 길이만 계산한다.
- 마운트되면 자동으로 duration_s 동안 수신한 뒤 결과를 1회 돌려준다.
- 마이크가 없거나 거부되면 {"status": "denied"/"unavailable"} 를 즉시 돌려준다.

반환값:
  {"status": "ok", "start_epoch_ms":..., "end_epoch_ms":..., "duration_s":...,
   "noise_floor_db":..., "threshold_db":..., "segments":[{"start_ms","end_ms"}...],
   "features": {...}, "quality": "ok"|"no_speech"|"noisy", "level_db_50hz":[...]}
연구용 시제품이며 진단 목적이 아니다.
"""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_component_func = components.declare_component(
    "voice_timing",
    path=_COMPONENT_DIR,
)


def voice_timing_component(duration_s: float = 16.0, key: str = None):
    """duration_s: 마운트 후 수신할 시간(초). 새로 시작할 때는 다른 key를 넘길 것."""
    return _component_func(duration_s=duration_s, key=key, default=None)
