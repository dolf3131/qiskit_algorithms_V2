from collections.abc import Sequence
from typing import Literal, Any

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
# EstimatorV2의 부모 클래스인 BaseEstimator를 임포트하여 더 유연하게 타입을 지정합니다.
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
# EstimatorGradientResult는 반환 타입 힌트를 위해 필요합니다.
from qiskit_algorithms.gradients import EstimatorGradientResult 


class FiniteDiffEstimatorGradientV2:
    """
    EstimatorV2에 기반하여 유한 차분법으로 기울기를 계산하는 클래스.
    이 클래스는 Qiskit의 BaseEstimatorGradient를 상속하지 않고 독립적으로 작동합니다.
    """
    def __init__(
        self,
        estimator: BaseEstimator, # EstimatorV2 (또는 BaseEstimatorV2를 상속하는 다른 Estimator) 객체를 받습니다.
        epsilon: float,
        *,
        method: Literal["central", "forward", "backward"] = "central",
    ):
        """
        Args:
            estimator: 기울기 계산에 사용될 EstimatorV2 (또는 BaseEstimatorV2 호환) 객체.
            epsilon: 유한 차분법을 위한 오프셋 크기. 양수여야 합니다.
            method: 기울기 계산 방법. 'central', 'forward', 'backward' 중 하나입니다.
        Raises:
            ValueError: epsilon이 양수가 아닐 경우.
            TypeError: method가 유효하지 않을 경우.
        """
        # Estimator 객체를 직접 할당합니다.
        self._estimator: BaseEstimator = estimator 
        
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._epsilon = epsilon
        
        if method not in ("central", "forward", "backward"):
            raise TypeError(
                f"The argument method should be central, forward, or backward: {method} is given."
            )
        self.method = method 

    # 이 클래스 인스턴스를 함수처럼 호출할 수 있도록 __call__ 메서드 구현
    def __call__(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]], # 각 circuit의 미분할 파라미터 리스트
        **options: Any, # 추가적인 Estimator 옵션을 받을 수 있도록 Any 타입 힌트 사용
    ) -> np.ndarray: 
        """
        주어진 회로들에 대한 Estimator 기울기를 계산합니다.

        Args:
            circuits: 기울기를 계산할 양자 회로 리스트.
            observables: 측정할 관측 가능량(연산자) 리스트.
            parameter_values: 회로에 바인딩할 매개변수 값 리스트.
            parameters: 미분할 특정 매개변수들의 시퀀스. 각 시퀀스는 해당 회로에 대응합니다.
            **options: Estimator 실행을 위한 추가 옵션.

        Returns:
            계산된 기울기(np.ndarray). 이 구현은 단일 회로/관측량 쌍에 대한 기울기를 반환합니다.

        Raises:
            RuntimeError: Estimator 작업이 실패할 경우.
            ValueError: 단일 회로/관측량 쌍에 대한 기울기를 기대했으나 여러 쌍이 반환될 경우.
        """
        
        all_pubs = [] # EstimatorV2.run()에 전달할 최종 PUB (Primitive Unified Bloc) 리스트

        # 여러 (circuit, observable, parameter_values, parameters) 쌍을 처리합니다.
        # VQE의 _get_evaluate_gradient는 단일 쌍을 전달하지만, 일반성을 위해 Sequence로 처리합니다.
        for circuit, observable, param_vals_list, params_to_diff in zip(
            circuits, observables, parameter_values, parameters
        ):
            # 미분할 파라미터들의 인덱스를 가져옵니다.
            indices = [circuit.parameters.data.index(p) for p in params_to_diff]
            
            # 오프셋 행렬 생성
            offset = np.identity(circuit.num_parameters)[indices, :]

            # EstimatorV2에 맞게 PUBs를 구성
            if self.method == "central":
                plus_values = np.asarray(param_vals_list) + self._epsilon * offset
                minus_values = np.asarray(param_vals_list) - self._epsilon * offset
                
                # 각 시프트된 값에 대한 PUB을 all_pubs에 추가합니다.
                for p_val in plus_values:
                    all_pubs.append((circuit, observable, p_val.tolist()))
                for m_val in minus_values:
                    all_pubs.append((circuit, observable, m_val.tolist()))
                
            elif self.method == "forward":
                plus_values = np.asarray(param_vals_list) + self._epsilon * offset
                all_pubs.append((circuit, observable, param_vals_list)) # 원본 값
                for p_val in plus_values:
                    all_pubs.append((circuit, observable, p_val.tolist()))

            elif self.method == "backward":
                minus_values = np.asarray(param_vals_list) - self._epsilon * offset
                all_pubs.append((circuit, observable, param_vals_list)) # 원본 값
                for m_val in minus_values:
                    all_pubs.append((circuit, observable, m_val.tolist()))
        
        # EstimatorV2.run()을 호출하고 결과를 받습니다.
        try:
            job = self._estimator.run(all_pubs, **options) # EstimatorV2.run()은 (pubs, **options) 시그니처
            results = job.result()
        except Exception as exc:
            raise RuntimeError("Estimator job for gradient failed.") from exc
        
        # 기울기 계산 및 포스트 프로세싱
        gradients_out = [] # 각 원래 circuit에 대한 최종 기울기 (np.ndarray)
        current_pub_idx = 0 # results.values 내에서 현재 처리 중인 PUB의 인덱스

        for circuit, observable, param_vals_list, params_to_diff in zip(
            circuits, observables, parameter_values, parameters
        ):
            indices = [circuit.parameters.data.index(p) for p in params_to_diff]
            
            if self.method == "central":
                n_pubs_for_this_circuit = 2 * len(indices)
                
                # 각 PUB 결과의 evs 값을 추출하여 넘파이 배열로 만듭니다.
                result_evs = np.array([results[current_pub_idx + k].data.evs
                                       for k in range(n_pubs_for_this_circuit)])
                
                gradient = (result_evs[: len(indices)] - result_evs[len(indices) :]) / (2 * self._epsilon)
                
            elif self.method == "forward":
                n_pubs_for_this_circuit = len(indices) + 1
                result_evs = np.array([results[current_pub_idx + k].data.evs
                                       for k in range(n_pubs_for_this_circuit)])
                gradient = (result_evs[1:] - result_evs[0]) / self._epsilon
                
            elif self.method == "backward":
                n_pubs_for_this_circuit = len(indices) + 1
                result_evs = np.array([results[current_pub_idx + k].data.evs
                                       for k in range(n_pubs_for_this_circuit)])
                gradient = (result_evs[0] - result_evs[1:]) / self._epsilon
            
            current_pub_idx += n_pubs_for_this_circuit # 다음 circuit의 PUB 시작점 업데이트
            
            gradients_out.append(gradient) # 최종 기울기 리스트에 추가

        # VQE의 _get_evaluate_gradient는 단일 np.ndarray 기울기를 기대합니다.
        # 이 __call__ 메서드는 VQE가 단일 (circuit, observable, parameter_values, parameters) 쌍을
        # 인자로 전달한다고 가정하므로, gradients_out 리스트는 항상 하나의 요소만 가질 것입니다.
        if len(gradients_out) == 1:
            return gradients_out[0] # 단일 np.ndarray 기울기 반환
        else:
            # 이 경우는 VQE의 호출 방식이 변경되거나 잘못되었을 때 발생합니다.
            raise ValueError(
                "FiniteDiffEstimatorGradientV2 expected gradients for a single circuit/observable pair, "
                f"but got {len(gradients_out)} pairs. Check the caller's logic."
            )