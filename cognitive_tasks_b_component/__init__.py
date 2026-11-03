# -*- coding: utf-8 -*-
"""
cognitive_tasks_b_component  (V2 인지과제 3~9번)

기획안의 나머지 과제를 한 컴포넌트에서 순서대로 실행한다.
  3 단어 외우기 -> 4 숫자 따라 누르기 -> 5 숫자·도형 번갈아 잇기 -> 6 동물 이름 말하기(음성 시간 정보만)
  -> 7 도형 짝 맞추기 -> 8 단어 회상(지연) -> 9 짧은 체크인(독자 문항, 예시)
모든 시간은 브라우저 performance.now() 기준이며, 소리는 크기 변화만 계산하고 저장·전사하지 않는다.
연구용 시제품이며 진단 목적이 아니다. 문항·기준은 임상 검증 전이다.
"""
import os
import streamlit.components.v1 as components

_DIR = os.path.dirname(os.path.abspath(__file__))
_func = components.declare_component("cognitive_tasks_b", path=_DIR)


def cognitive_tasks_b_component(word_set: str = "", time_scale: float = 1.0,
                                debug: bool = False, key: str = None):
    """word_set: 'A'|'B'|'C' (빈 값이면 무작위). time_scale/debug는 시험용."""
    return _func(word_set=word_set, time_scale=time_scale, debug=debug,
                 key=key, default=None)
