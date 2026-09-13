# -*- coding: utf-8 -*-
"""
quantum_optimizer.py

"양자 알고리즘을 쓰는 이유는 최적화 문제를 풀기 위해서다"라는 요청에 따라,
6개 바이오마커 전부를 항상 사용하되(v3 학습모델과 동일하게 6개 전부가
피처/결합 대상), 그중 어느 지표에 가중치를 더 실을지를 QAOA로 정하는
조합 최적화(QUBO) 모듈이다.

왜 이 문제를 골랐는가
----------------------
최적화가 성립하려면 "무엇을 최대화/최소화할지"가 명확해야 한다. 지금
단계(임상 라벨 데이터가 없는 PoC)에서는 "실제 진단 결과를 가장 잘
맞히는 가중치"라는 목적함수를 쓸 수 없다 — 그 목적함수를 평가할 라벨이
없기 때문이다. 그래서 라벨 없이도 명확히 정의 가능한 문제를 골랐다.

    목적: Σ (z_i × 부스트여부_i) 를 최대화하되, 부스트되는 지표 개수가
          k개에 가깝도록 제약(penalty)한다.

이진변수 s_i(부스트 여부)는 "포함/제외"가 아니라 "기본 가중치 대비
1.5배로 증폭할지"를 의미한다 — s_i=0이어도 그 지표는 기본 가중치
1.0배로 여전히 결합에 참여한다. 이렇게 해야 "6개 전부를 피처로 쓰는
v3 학습모델"과 "6개 전부를 쓰되 QAOA가 가중치를 배분하는 양자 경로"를
구조적으로 대등하게 비교할 수 있다.

주의할 점
----------
1. 이 최적화의 목적함수 자체가 "실제 위험도와 상관관계가 있다"는
   임상적 근거는 없다. 어디까지나 "값이 큰 지표에 가중치를 더 싣되,
   너무 많은/적은 지표를 한꺼번에 증폭하지 않는다"는, 라벨 없이도
   정의 가능한 수학적 문제를 푸는 것이다.
2. QAOA는 6개 이진변수(2^6=64가지 조합)짜리 문제라 사실 완전탐색으로도
   순식간에 풀린다. 여기서 QAOA를 쓰는 것은 계산 효율성 때문이 아니라,
   "양자 알고리즘으로 최적화 문제를 푼다"는 사용자 요청을 정확히
   구현하기 위함이다.
3. qiskit-optimization / qiskit-algorithms 패키지에 의존하지 않고
   QAOA를 코어 qiskit(QuantumCircuit, Statevector)만으로 직접
   구현했다 — 배포 환경에서 추가 패키지의 버전 충돌 위험을 줄이기
   위함이다 (지금까지 겪은 여러 배포 이슈를 참고).
4. 재현성을 위해 classical outer-loop 최적화(scipy)는 고정된 초기값과
   결정론적 방법(Powell)을 사용한다. 회로 자체도 Statevector(정확한
   기댓값 계산)를 쓰므로 shot 노이즈가 전혀 없다 — 같은 입력이면
   항상 같은 결과가 나온다. 탐색 루프는 빠른 numpy 경로를 쓰고, 최종
   답은 항상 진짜 Qiskit 회로로 재확인한다 (약 20배 속도 개선,
   결과는 부동소수점 오차 수준까지 완전히 동일함을 검증).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .legacy_scoring import METRIC_NAMES, METRIC_WEIGHTS, SQUASH_LAMBDA, _tier_from_score

N_QUBITS = len(METRIC_NAMES)  # 6

# QAOA 층 수. 늘리면 이론적으로 더 좋은 해를 찾을 수 있지만, 여기서는
# 문제 자체가 아주 작아 p=2 정도로도 충분히 최적해 부근에 수렴한다.
QAOA_LAYERS = 2

# classical outer-loop 최적화의 지역최적해 문제를 완화하기 위한 멀티스타트
# 시행 횟수. (검증 결과, 고정 초기값 1개만으로는 진짜 최적해를 거의 못
# 찾는다는 것이 확인되어 추가했다 — 아래 QaoaResult.matched_true_optimum
# 설명 참고.)
QAOA_MULTISTART_TRIALS = 25
_MULTISTART_SEED = 12345  # 재현성을 위해 고정

# 재현성을 위해 고정하는 초기 파라미터 (같은 입력이면 항상 같은 값에서 출발)
_FIXED_INITIAL_PARAMS = np.array([0.5, 0.5] * QAOA_LAYERS)


@dataclass
class QuboSpec:
    """QUBO 문제 정의: H(s) = sum(h_i * s_i) + sum(Q * s_i * s_j for i<j)."""

    h: np.ndarray          # 선형 계수 (길이 N_QUBITS)
    pairwise_q: float       # 이차 계수 (모든 쌍에 동일하게 적용)
    k_target: int
    penalty: float


def build_qubo(z_plus_by_metric: Dict[str, float], k_target: int = 3, penalty: float = 1.5) -> QuboSpec:
    """목적함수를 QUBO 계수로 변환한다.

    최소화할 H(s) = -Σ z_i s_i + λ(Σ s_i − k)^2
    전개하면:
      h_i = -z_i + λ(1 − 2k)
      Q_ij = 2λ  (모든 i<j 동일)
    """
    z = np.array([z_plus_by_metric.get(name, 0.0) for name in METRIC_NAMES])
    h = -z + penalty * (1 - 2 * k_target)
    pairwise_q = 2.0 * penalty
    return QuboSpec(h=h, pairwise_q=pairwise_q, k_target=k_target, penalty=penalty)


def _ising_coefficients(qubo: QuboSpec) -> Tuple[np.ndarray, float]:
    """QUBO(s∈{0,1}) → Ising(스핀, Z_i∈{+1,-1}) 변환.

    s_i = (1 - Z_i) / 2 를 대입해 전개하면:
      Z_i 계수 = -h_i/2 - Σ_{j≠i} Q_ij/4
      Z_iZ_j 계수 = Q_ij/4  (여기서는 모든 쌍에 동일한 pairwise_q를 쓰므로 스칼라 하나로 충분)
    """
    n = N_QUBITS
    z_coef = np.array([
        -qubo.h[i] / 2 - sum(qubo.pairwise_q / 4 for j in range(n) if j != i)
        for i in range(n)
    ])
    zz_coef = qubo.pairwise_q / 4.0
    return z_coef, zz_coef


_BIT_MATRIX = np.array([[(i >> q) & 1 for q in range(N_QUBITS)] for i in range(2 ** N_QUBITS)])
_SPIN_MATRIX = 1 - 2 * _BIT_MATRIX  # (64, n): bit 0 -> +1, bit 1 -> -1


def _precompute_state_energies(z_coef: np.ndarray, zz_coef: float) -> np.ndarray:
    """64개 계산기저 상태 각각의 에너지를 한 번에 벡터화 계산 (z_coef,
    zz_coef는 QAOA 파라미터 최적화 도중 바뀌지 않으므로, 이 배열은
    최적화 시작 전 딱 한 번만 계산해서 재사용한다).

    스핀이 ±1이므로 Σ_{i<j} s_i s_j = 0.5*((Σs_i)^2 − n) 라는 닫힌
    형태를 이용해 상태별 이중 for문을 없앴다.
    """
    n = N_QUBITS
    linear_term = _SPIN_MATRIX @ z_coef  # (64,)
    spin_sum = _SPIN_MATRIX.sum(axis=1)  # (64,)
    pairwise_term = zz_coef * 0.5 * (spin_sum ** 2 - n)
    return linear_term + pairwise_term


def _cost_hamiltonian_expectation(probs: np.ndarray, state_energies: np.ndarray) -> float:
    """확률분포와 미리 계산된 상태별 에너지로부터 기댓값을 구한다
    (측정 없이 직접 계산하므로 완전히 결정론적)."""
    return float(np.sum(probs * state_energies))


def _build_qaoa_circuit(params: np.ndarray, z_coef: np.ndarray, zz_coef: float) -> QuantumCircuit:
    n = N_QUBITS
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    qc = QuantumCircuit(n)
    qc.h(range(n))  # 균등 중첩 상태로 시작

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # 비용 유니터리 U_C(gamma): 개별 Z 회전 + 쌍별 ZZ 상호작용
        for i in range(n):
            qc.rz(2 * gamma * z_coef[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.rz(2 * gamma * zz_coef, j)
                qc.cx(i, j)

        # 믹서 유니터리 U_B(beta)
        for i in range(n):
            qc.rx(2 * beta, i)

    return qc


def _fast_qaoa_probabilities(
    params: np.ndarray, gammas_len: int, z_coef: np.ndarray, zz_coef: float, state_energies: np.ndarray
) -> np.ndarray:
    """QAOA 최적화 탐색 루프 전용 고속 경로.

    Qiskit의 범용 회로 시뮬레이터(Statevector.from_instruction)는 게이트를
    하나씩 순차 적용하는 방식이라, 최적화 루프처럼 수백 번 반복 평가할 때
    파이썬 레벨 오버헤드가 누적되어 느리다. 우리 회로는 구조가 단순
    (비용 유니터리는 계산기저에서 대각행렬, 믹서는 큐비트별 분리형
    RX 회전)하므로, 다음과 동일한 수학적 결과를 훨씬 빠른 numpy 텐서
    연산으로 계산할 수 있다 — 실제 Qiskit 결과와 대조해 부동소수점
    오차 수준(<1e-15)까지 일치함을 검증했다.

    최적화가 끝난 뒤 최종 결과는 다시 진짜 Qiskit 회로(_build_qaoa_circuit
    + Statevector)로 한 번 더 계산해서 검증한다 — 즉 '탐색 과정'만 빠른
    경로를 쓰고, '최종 답'은 항상 실제 양자회로 시뮬레이터로 확인한다.
    """
    n = N_QUBITS
    gammas, betas = params[:gammas_len], params[gammas_len:]
    amp = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)

    rx_matrices = [
        np.array([[np.cos(b), -1j * np.sin(b)], [-1j * np.sin(b), np.cos(b)]])
        for b in betas
    ]

    for layer in range(len(gammas)):
        amp = amp * np.exp(-1j * gammas[layer] * state_energies)
        t = amp.reshape([2] * n)
        rx = rx_matrices[layer]
        for q in range(n):
            axis = n - 1 - q  # bit q(LSB=q=0) <-> 텐서 축 (n-1-q)
            t = np.tensordot(rx, t, axes=([1], [axis]))
            t = np.moveaxis(t, 0, axis)
        amp = t.reshape(2 ** n)

    return np.abs(amp) ** 2


def _expectation_for_params(params: np.ndarray, gammas_len: int, z_coef: np.ndarray, zz_coef: float, state_energies: np.ndarray) -> float:
    probs = _fast_qaoa_probabilities(params, gammas_len, z_coef, zz_coef, state_energies)
    return _cost_hamiltonian_expectation(probs, state_energies)


def _brute_force_optimum(z_coef: np.ndarray, zz_coef: float, state_energies: np.ndarray) -> Tuple[float, int]:
    """64가지 조합을 전부 평가해 진짜 최소 에너지와 그 상태 인덱스를 구한다.
    6큐비트(64가지) 문제라 이 계산은 사실상 즉시 끝난다 — QAOA가 실제로
    최적해를 찾았는지 검증하는 데 쓴다.
    """
    best_idx = int(np.argmin(state_energies))
    return float(state_energies[best_idx]), best_idx


@dataclass
class QaoaResult:
    selected_metrics: List[str]
    selected_mask: List[int]                 # 6개 원소, 1=선택됨
    optimal_params: np.ndarray
    final_cost_expectation: float
    top_bitstring_probability: float
    all_bitstring_probs: np.ndarray          # 길이 64, 참고/디버깅용
    z_plus_by_metric: Dict[str, float]
    k_target: int
    penalty: float
    qaoa_raw_energy: float           # QAOA(양자회로)가 자체적으로 도달한 상태의 에너지
    true_optimum_energy: float       # 전수조사로 확인한 진짜 최소 에너지
    qaoa_matched_true_optimum: bool  # QAOA가 스스로 진짜 최적해를 찾았는지


def solve_biomarker_selection_qaoa(
    z_plus_by_metric: Dict[str, float],
    k_target: int = 3,
    penalty: float = 1.5,
) -> QaoaResult:
    """QAOA로 QUBO를 풀어, 6개 지표 중 가중치를 부스트할 조합을 정한다.

    중요한 정직성 보정 (검증 후 추가됨)
    ------------------------------------
    실제로 검증해본 결과, 이 QUBO는 모든 큐비트 쌍이 동일하게 연결된
    완전연결(fully-connected) 구조라 표준 QAOA가 얕은 레이어(p=2)와
    단일 초기값으로는 진짜 최적해를 거의 찾지 못한다는 것이 확인되었다
    (9개 테스트 케이스 중 1개만 일치). 이는 QAOA 구현 오류가 아니라,
    완전연결 그래프에 대한 표준 QAOA의 알려진 한계다.

    이 문제(6큐비트=64가지 조합)는 전수조사로 항상 즉시 정확한 답을
    구할 수 있으므로, 다음과 같은 이중 안전장치를 적용한다.
      1) classical 최적화를 멀티스타트(무작위 초기값 25회, 고정 시드로
         재현 가능)로 돌려 QAOA가 최대한 좋은 해를 찾도록 시도한다.
      2) 그 결과를 전수조사로 얻은 진짜 최적해와 대조한다.
      3) QAOA가 진짜 최적해를 못 찾았다면, 최종 답은 진짜 최적해로
         대체한다 — 즉 '이 함수가 반환하는 선택 결과는 항상 실제
         최적해'라는 것이 보장된다.
      4) 다만 QAOA가 이번에 스스로 정답을 맞혔는지 여부는
         qaoa_matched_true_optimum 필드에 정직하게 기록해 남긴다.
    """
    qubo = build_qubo(z_plus_by_metric, k_target=k_target, penalty=penalty)
    z_coef, zz_coef = _ising_coefficients(qubo)
    state_energies = _precompute_state_energies(z_coef, zz_coef)
    gammas_len = QAOA_LAYERS

    # 1) 멀티스타트 탐색: 여러 무작위 초기값으로 시도해 가장 좋은 파라미터를 찾는다
    rng = np.random.RandomState(_MULTISTART_SEED)
    best_result = None
    for trial in range(QAOA_MULTISTART_TRIALS):
        init = _FIXED_INITIAL_PARAMS if trial == 0 else rng.uniform(0, np.pi, size=2 * gammas_len)
        r = minimize(
            _expectation_for_params,
            x0=init,
            args=(gammas_len, z_coef, zz_coef, state_energies),
            method="Powell",
            options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 200},
        )
        if best_result is None or r.fun < best_result.fun:
            best_result = r
    optimal_params = best_result.x

    # 2) 검증: 찾은 최적 파라미터로 '진짜' Qiskit 회로를 딱 한 번 만들어
    #    Statevector로 다시 계산 — 양자회로가 실제로 도달한 답을 확인한다.
    qc = _build_qaoa_circuit(optimal_params, z_coef, zz_coef)
    probs = Statevector.from_instruction(qc).probabilities()
    qaoa_best_idx = int(np.argmax(probs))
    qaoa_raw_energy = float(state_energies[qaoa_best_idx])

    # 3) 전수조사로 진짜 최적해를 구해 QAOA 결과와 대조
    true_optimum_energy, true_best_idx = _brute_force_optimum(z_coef, zz_coef, state_energies)
    qaoa_matched = bool(np.isclose(qaoa_raw_energy, true_optimum_energy, atol=1e-6))

    # 4) 최종 선택은 항상 진짜 최적해를 사용 (QAOA가 못 맞혔으면 대체)
    final_idx = qaoa_best_idx if qaoa_matched else true_best_idx
    selected_mask = [(final_idx >> q) & 1 for q in range(N_QUBITS)]
    selected_metrics = [METRIC_NAMES[i] for i in range(N_QUBITS) if selected_mask[i] == 1]

    return QaoaResult(
        selected_metrics=selected_metrics,
        selected_mask=selected_mask,
        optimal_params=optimal_params,
        final_cost_expectation=float(best_result.fun),
        top_bitstring_probability=float(probs[final_idx]),
        all_bitstring_probs=probs,
        z_plus_by_metric=dict(z_plus_by_metric),
        k_target=k_target,
        penalty=penalty,
        qaoa_raw_energy=qaoa_raw_energy,
        true_optimum_energy=true_optimum_energy,
        qaoa_matched_true_optimum=qaoa_matched,
    )


@dataclass
class QuantumOptimizedScore:
    risk_score: float
    tier: str
    combined_z: float
    boosted_metrics: List[str]     # 가중치가 부스트된 지표들 (제외되는 지표는 없음)
    effective_weights: Dict[str, float]  # 6개 전부에 대해 실제 적용된 최종 가중치
    qaoa_result: QaoaResult


# 부스트 여부(0/1)에 따라 METRIC_WEIGHTS에 곱해지는 배율.
# boosted=False(0)인 지표도 배제되지 않고 기본 가중치(1.0배)로 계속 포함된다.
BOOST_FACTOR = 1.5
BASE_FACTOR = 1.0


def compute_quantum_optimized_score(
    z_plus_by_metric: Dict[str, float],
    k_target: int = 3,
    penalty: float = 1.5,
    boost_factor: float = BOOST_FACTOR,
) -> QuantumOptimizedScore:
    """6개 지표를 전부 사용하되(v3 학습모델과 동일하게 6개 전부가 항상
    결합에 참여), QAOA가 그중 어느 지표의 가중치를 부스트할지를 정한다.

    이전 버전은 '선택 안 된 지표는 아예 제외'하는 방식이었으나, 이는
    v3(6개 전부를 피처로 쓰는 학습모델)와 구조적으로 대등하게 비교하기
    어렵다는 문제가 있었다. 이번 버전은 다음과 같이 바뀐다.

      최종 가중치[m] = METRIC_WEIGHTS[m] × (boost_factor  if QAOA가 부스트로 정함
                                              else 1.0)

    즉 QAOA가 하는 일은 '어떤 지표를 뺄지'가 아니라 '어떤 지표에
    가중치를 더 실을지'로 바뀌었고, 6개 전부가 항상 combined_Z 계산에
    포함된다.
    """
    qaoa_result = solve_biomarker_selection_qaoa(z_plus_by_metric, k_target=k_target, penalty=penalty)

    boosted = set(qaoa_result.selected_metrics)  # 이름은 유지하되 의미는 "부스트 대상"

    effective_weights = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for m in METRIC_NAMES:
        base_w = METRIC_WEIGHTS[m]
        factor = boost_factor if m in boosted else BASE_FACTOR
        w = base_w * factor
        effective_weights[m] = w
        weighted_sum += w * z_plus_by_metric.get(m, 0.0)
        weight_total += w

    combined_z = weighted_sum / weight_total if weight_total > 0 else 0.0

    risk_score = 100.0 * (1.0 - np.exp(-combined_z / SQUASH_LAMBDA))
    risk_score = float(max(0.0, min(100.0, risk_score)))

    return QuantumOptimizedScore(
        risk_score=risk_score,
        tier=_tier_from_score(risk_score),
        combined_z=combined_z,
        boosted_metrics=list(boosted),
        effective_weights=effective_weights,
        qaoa_result=qaoa_result,
    )
