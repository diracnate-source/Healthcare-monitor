# -*- coding: utf-8 -*-
"""
quantum_modifier.py

실제 임상 근거는 아니지만, 사용자가 명시적으로 요청한 "양자 회로 시뮬레이션
라이브러리(Qiskit)를 위험도 계산에 활용"을 안전하고 재현 가능한 형태로
구현한 실험적(EXPERIMENTAL) 모듈이다.

중요한 전제 (반드시 읽을 것)
----------------------------
1. 이 모듈은 실제 양자 하드웨어를 쓰지 않는다. Statevector 계열 함수는
   순수 클래식 컴퓨터에서 양자역학 방정식을 그대로 풀어 확률분포를
   계산하는 것으로, 6큐비트(2^6=64차원) 정도는 행렬 연산으로 직접 풀어도
   완전히 동일한 결과가 나온다.
2. risk_score에 실제로 반영되는 보정치(apply_quantum_modifier)는 여전히
   ±QUANTUM_MODIFIER_MAX_POINTS로 제한되고 기본 비활성화 상태이며,
   결정론적(Statevector) 계산만 사용한다 — 이 부분은 이전 버전과 동일한
   원칙을 유지한다.
3. 이번에 추가된 세 가지 기능(얽힘 시각화, 양자지표 강조, 노이즈/샷
   변동성 패널)은 아래와 같은 명확한 역할 분리 원칙 아래 구현했다.

   - 얽힘 시각화(compute_entanglement_analysis / render_entanglement_figure):
     "고전적 독립 결합(단순 가중합)과 양자 얽힘이 실제로 다른 상호작용을
     만든다"는 것을, 큐비트별 측정 확률(marginal)과 인접 큐비트 간
     상관계수를 얽힘 전(product state)/후(entangled state)로 비교해
     정량적으로 보여준다. 이 비교 자체는 100% 결정론적이고 실제 회로
     계산 결과이며, 다만 그 차이가 '치매 위험도와 상관관계가 있다'는
     임상적 의미는 없다 — 두 종류의 수학적 결합 방식이 다르다는 것만
     보여준다.

   - 양자지표 강조(QuantumModifierResult에 담긴 entropy/concentration):
     리포트에서 비중 있게(크게, 눈에 띄게) 보여주도록 렌더링 함수를
     제공하지만, 절대 '질환 판정의 근거'라고 표기하지 않는다. 항상
     "실험적 지표— 새로운 임상 정보를 추가하지 않음"이라는 라벨을
     동반한다. risk_score 계산에 대한 실제 영향력도 여전히
     ±QUANTUM_MODIFIER_MAX_POINTS로 제한된다.

   - 노이즈/샷 변동성 패널(simulate_shot_noise_variability /
     render_noise_variability_figure): 실제 양자 하드웨어(또는 샷 기반
     샘플링)를 썼다면 결과가 얼마나 흔들렸을지를 별도로 시뮬레이션해
     '교육용'으로만 보여준다. 이 함수의 출력은 risk_score 계산 경로
     (apply_quantum_modifier)에 전혀 연결되지 않는다 — 실제 판정 결과가
     매 세션 무작위로 바뀌는 일은 없다.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .legacy_scoring import METRIC_NAMES

# ============================================================
# 1) risk_score에 실제로 반영되는 부분 (기존과 동일한 안전 원칙)
# ============================================================

ENABLE_QUANTUM_MODIFIER_DEFAULT = False
QUANTUM_MODIFIER_MAX_POINTS = 5.0
Z_SATURATION = 4.0


@dataclass
class QuantumModifierResult:
    enabled: bool
    entropy: float
    max_probability: float
    modifier_points: float
    note: str


def _build_circuit(z_plus_by_metric: Dict[str, float], entangle: bool = True) -> QuantumCircuit:
    """entangle=False이면 CNOT을 걸지 않아 큐비트들이 서로 독립인 곱 상태
    (product state)로 남는다 — '얽힘이 없을 때'와 비교하기 위한 기준선."""
    qc = QuantumCircuit(len(METRIC_NAMES))
    for i, name in enumerate(METRIC_NAMES):
        z = z_plus_by_metric.get(name, 0.0)
        z_clamped = max(0.0, min(z, Z_SATURATION))
        angle = (z_clamped / Z_SATURATION) * math.pi
        qc.rx(angle, i)
    if entangle:
        for i in range(len(METRIC_NAMES) - 1):
            qc.cx(i, i + 1)
    return qc


def compute_quantum_modifier(
    z_plus_by_metric: Dict[str, float],
    enabled: bool = ENABLE_QUANTUM_MODIFIER_DEFAULT,
) -> QuantumModifierResult:
    if not enabled:
        return QuantumModifierResult(
            enabled=False, entropy=0.0, max_probability=1.0,
            modifier_points=0.0,
            note="양자회로 보정 비활성화 상태 (기본값). risk_score에 영향 없음.",
        )

    qc = _build_circuit(z_plus_by_metric, entangle=True)
    probs = Statevector.from_instruction(qc).probabilities()

    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    max_probability = float(np.max(probs))

    n_qubits = len(METRIC_NAMES)
    max_entropy = n_qubits
    normalized = (entropy / max_entropy) * 2.0 - 1.0
    modifier_points = normalized * QUANTUM_MODIFIER_MAX_POINTS

    return QuantumModifierResult(
        enabled=True,
        entropy=entropy,
        max_probability=max_probability,
        modifier_points=modifier_points,
        note=(
            "실험적 지표: 이미 계산된 6개 z-score를 양자 회로 시뮬레이터(Statevector, "
            "결정론적)에 인코딩해 얻은 엔트로피 기반 값입니다. 새로운 임상적 정보를 "
            "추가하지 않으며, 질환 판정의 근거가 아닙니다."
        ),
    )


def apply_quantum_modifier(
    data_driven_result: Dict,
    enabled: bool = ENABLE_QUANTUM_MODIFIER_DEFAULT,
) -> Dict:
    from .legacy_scoring import _tier_from_score

    z_plus_by_metric = {
        name: entry["z_plus"]
        for name, entry in data_driven_result.get("detail", {}).items()
    }

    q_result = compute_quantum_modifier(z_plus_by_metric, enabled=enabled)

    result = dict(data_driven_result)
    if q_result.enabled:
        adjusted_score = data_driven_result["risk_score"] + q_result.modifier_points
        adjusted_score = max(0.0, min(100.0, adjusted_score))
        result["risk_score"] = adjusted_score
        result["tier"] = _tier_from_score(adjusted_score)

    result["quantum_modifier"] = {
        "enabled": q_result.enabled,
        "entropy": q_result.entropy,
        "max_probability": q_result.max_probability,
        "modifier_points": q_result.modifier_points,
        "note": q_result.note,
    }
    return result


# ============================================================
# 2) 얽힘(Entanglement) 시각화 — 시각화 전용, risk_score와 무관
# ============================================================

@dataclass
class EntanglementAnalysis:
    metric_names: List[str]
    marginals_product: np.ndarray       # 얽힘 없을 때, 큐비트별 P(측정값=1)
    marginals_entangled: np.ndarray     # 얽힘 있을 때, 큐비트별 P(측정값=1)
    pair_correlation_product: np.ndarray    # 인접 큐비트 쌍의 상관계수 (얽힘 전, 전부 0에 가까움)
    pair_correlation_entangled: np.ndarray  # 인접 큐비트 쌍의 상관계수 (얽힘 후)
    entropy_product: float
    entropy_entangled: float


def _marginals_and_pair_correlation(probs: np.ndarray, n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """전체 상태(2^n개 확률)로부터 (a) 큐비트별 P(값=1), (b) 인접 큐비트
    쌍의 상관계수를 계산한다.

    상관계수 = P(i=1, i+1=1) - P(i=1)*P(i+1=1)
    독립(곱 상태)이면 항상 0, 얽혀 있으면 0이 아닌 값을 가진다 —
    이게 바로 "고전적 단순 결합과 다른 비선형 상호작용"의 정량적 증거.
    """
    n_states = 2 ** n_qubits
    marginals = np.zeros(n_qubits)
    for state_idx in range(n_states):
        p = probs[state_idx]
        if p <= 0:
            continue
        for q in range(n_qubits):
            bit = (state_idx >> q) & 1
            if bit == 1:
                marginals[q] += p

    pair_corr = np.zeros(n_qubits - 1)
    for q in range(n_qubits - 1):
        joint11 = 0.0
        for state_idx in range(n_states):
            p = probs[state_idx]
            if p <= 0:
                continue
            bit_a = (state_idx >> q) & 1
            bit_b = (state_idx >> (q + 1)) & 1
            if bit_a == 1 and bit_b == 1:
                joint11 += p
        pair_corr[q] = joint11 - marginals[q] * marginals[q + 1]

    return marginals, pair_corr


def compute_entanglement_analysis(z_plus_by_metric: Dict[str, float]) -> EntanglementAnalysis:
    """얽힘이 없는 회로(product state)와 얽힘이 있는 회로(entangled state)를
    나란히 계산해, 큐비트별 측정확률과 인접 큐비트 간 상관계수 차이를 만든다.
    전부 Statevector 기반 결정론적 계산이다.
    """
    n_qubits = len(METRIC_NAMES)

    qc_product = _build_circuit(z_plus_by_metric, entangle=False)
    probs_product = Statevector.from_instruction(qc_product).probabilities()

    qc_entangled = _build_circuit(z_plus_by_metric, entangle=True)
    probs_entangled = Statevector.from_instruction(qc_entangled).probabilities()

    marg_p, corr_p = _marginals_and_pair_correlation(probs_product, n_qubits)
    marg_e, corr_e = _marginals_and_pair_correlation(probs_entangled, n_qubits)

    entropy_p = float(-np.sum(probs_product * np.log2(probs_product + 1e-12)))
    entropy_e = float(-np.sum(probs_entangled * np.log2(probs_entangled + 1e-12)))

    return EntanglementAnalysis(
        metric_names=list(METRIC_NAMES),
        marginals_product=marg_p,
        marginals_entangled=marg_e,
        pair_correlation_product=corr_p,
        pair_correlation_entangled=corr_e,
        entropy_product=entropy_p,
        entropy_entangled=entropy_e,
    )


def render_entanglement_figure(analysis: EntanglementAnalysis):
    """큐비트별 측정확률(marginal) 비교 + 인접 큐비트 상관계수 비교를
    하나의 matplotlib Figure로 렌더링한다. (한글 폰트 의존성을 피하기 위해
    축·범례는 영문/숫자만 사용 — 한국어 설명은 호출 측(Streamlit)에서
    caption으로 감싸는 것을 권장한다.)
    """
    n = len(analysis.metric_names)
    qubit_labels = [f"Q{i}" for i in range(n)]
    pair_labels = [f"Q{i}-Q{i+1}" for i in range(n - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    x = np.arange(n)
    width = 0.35
    ax.bar(x - width / 2, analysis.marginals_product, width, label="No entanglement (product state)", color="#95a5a6")
    ax.bar(x + width / 2, analysis.marginals_entangled, width, label="With entanglement (CNOT chain)", color="#8e44ad")
    ax.set_xticks(x)
    ax.set_xticklabels(qubit_labels)
    ax.set_ylabel("P(measured = 1)")
    ax.set_title("Per-qubit marginal probability")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper right")

    ax2 = axes[1]
    x2 = np.arange(n - 1)
    ax2.bar(x2 - width / 2, analysis.pair_correlation_product, width, label="No entanglement", color="#95a5a6")
    ax2.bar(x2 + width / 2, analysis.pair_correlation_entangled, width, label="With entanglement", color="#8e44ad")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(pair_labels)
    ax2.set_ylabel("Pairwise correlation")
    ax2.set_title("Adjacent-qubit correlation\n(0 = statistically independent)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig


# ============================================================
# 3) 노이즈 / 샷(shot) 변동성 — 교육용 패널, risk_score와 완전히 분리
# ============================================================

@dataclass
class NoiseVariabilityResult:
    shot_counts: List[int]
    noiseless_entropy: float                    # Statevector 기준(참값)
    entropy_samples_by_shots: Dict[int, List[float]]  # shot 수별 반복 시행 엔트로피 추정치들
    depolarizing_prob: float
    trials_per_shot_count: int


def simulate_shot_noise_variability(
    z_plus_by_metric: Dict[str, float],
    shot_counts: Tuple[int, ...] = (64, 256, 1024, 4096),
    depolarizing_prob: float = 0.03,
    trials_per_shot_count: int = 30,
) -> NoiseVariabilityResult:
    """실제 양자 하드웨어(또는 샷 기반 샘플링)를 쓴다면 결과가 얼마나
    흔들리는지를 보여주기 위한 교육용 시뮬레이션.

    - depolarizing_prob: 게이트마다 일정 확률로 상태가 무작위로 흐트러지는
      가장 기본적인 노이즈 모델(depolarizing noise)의 강도.
    - shot_counts가 작을수록(=측정을 적게 반복할수록) 추정치의 분산이 커짐을
      보여주는 것이 목적.
    - 이 함수의 반환값은 risk_score 계산에 전혀 쓰이지 않는다. 순수하게
      "만약 노이즈가 있었다면"을 보여주는 참고 자료다.
    """
    qc = _build_circuit(z_plus_by_metric, entangle=True)
    qc_measured = qc.copy()
    qc_measured.measure_all()

    ideal_probs = Statevector.from_instruction(qc).probabilities()
    noiseless_entropy = float(-np.sum(ideal_probs * np.log2(ideal_probs + 1e-12)))

    noise_model = NoiseModel()
    err1 = depolarizing_error(depolarizing_prob, 1)
    err2 = depolarizing_error(depolarizing_prob, 2)
    noise_model.add_all_qubit_quantum_error(err1, ["rx"])
    noise_model.add_all_qubit_quantum_error(err2, ["cx"])

    sim = AerSimulator(noise_model=noise_model)
    transpiled = transpile(qc_measured, sim)

    entropy_samples_by_shots: Dict[int, List[float]] = {}

    for shots in shot_counts:
        entropies = []
        for trial in range(trials_per_shot_count):
            result = sim.run(transpiled, shots=shots, seed_simulator=trial).result()
            counts = result.get_counts()
            total = sum(counts.values())
            probs_emp = np.array([c / total for c in counts.values()])
            ent = float(-np.sum(probs_emp * np.log2(probs_emp + 1e-12)))
            entropies.append(ent)
        entropy_samples_by_shots[shots] = entropies

    return NoiseVariabilityResult(
        shot_counts=list(shot_counts),
        noiseless_entropy=noiseless_entropy,
        entropy_samples_by_shots=entropy_samples_by_shots,
        depolarizing_prob=depolarizing_prob,
        trials_per_shot_count=trials_per_shot_count,
    )


def render_noise_variability_figure(result: NoiseVariabilityResult):
    """shot 수가 늘어날수록 엔트로피 추정치가 참값(noiseless_entropy)
    주위로 얼마나 좁게 모이는지를 boxplot으로 보여준다."""
    fig, ax = plt.subplots(figsize=(7, 4.2))

    data = [result.entropy_samples_by_shots[s] for s in result.shot_counts]
    positions = np.arange(len(result.shot_counts))

    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#f39c12")
        box.set_alpha(0.6)

    ax.axhline(
        result.noiseless_entropy, color="#2c3e50", linestyle="--", linewidth=1.5,
        label=f"Noiseless (statevector) entropy = {result.noiseless_entropy:.3f}",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(s) for s in result.shot_counts])
    ax.set_xlabel("Number of shots (measurement repetitions)")
    ax.set_ylabel("Estimated entropy")
    ax.set_title(
        f"Shot-noise variability under a simple depolarizing noise model\n"
        f"(gate error rate = {result.depolarizing_prob:.0%}, "
        f"{result.trials_per_shot_count} trials per shot count)"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig
