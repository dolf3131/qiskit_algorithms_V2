from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2 # EstimatorV2의 부모 클래스
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Qiskit Algorithms의 옵티마이저 추상 클래스
from qiskit_algorithms.optimizers import Optimizer, Minimizer
from qiskit_algorithms.optimizers import OptimizerResult # SciPy OptimizeResult 호환

import numpy as np
from typing import Callable, Any, List

from time import time
import logging
logger = logging.getLogger(__name__)

# Qiskit Algorithms의 기본 알고리즘 클래스 및 VQE 결과 타입
from qiskit_algorithms import VariationalAlgorithm, MinimumEigensolver
from qiskit_algorithms.minimum_eigensolvers import VQEResult

# 외부 파일에서 필요한 클래스들을 임포트합니다.
# 이 클래스들이 정의된 파일 이름을 확인하고, 필요에 따라 임포트 경로를 수정하세요.
from finite_estimator_gradient_v2 import FiniteDiffEstimatorGradientV2

# VQE 알고리즘의 사용자 정의 구현
class VariationalQuantumEigensolverV2(VariationalAlgorithm, MinimumEigensolver):
    """
    Qiskit의 `EstimatorV2` 프리미티브에 맞춰 재구현된 Variational Quantum Eigensolver (VQE) 알고리즘입니다.
    사용자 정의 옵티마이저 및 기울기 계산기와 함께 작동합니다.
    """
    def __init__(
        self,
        estimator: BaseEstimatorV2,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer,
        *,
        gradient: FiniteDiffEstimatorGradientV2 | None = None, # 사용자 정의 기울기 계산기 타입 힌트
        initial_point: np.ndarray | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
    ) -> None:
        """
        Args:
            estimator: 기대값 계산에 사용될 EstimatorV2 프리미티브.
            ansatz: 시도 상태를 준비하는 매개변수화된 양자 회로.
            optimizer: 최소 에너지를 찾는 클래식 옵티마이저. Qiskit Optimizer 또는 Minimizer 프로토콜을 따르는 Callable.
            gradient: (선택 사항) 옵티마이저와 함께 사용될 Estimator 기울기.
            initial_point: (선택 사항) 옵티마이저를 위한 초기 매개변수 값.
            callback: (선택 사항) 각 최적화 스텝에서 중간 데이터에 접근할 수 있는 콜백 함수.
        """
        super().__init__()

        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.gradient = gradient
        # VariationalAlgorithm 인터페이스의 getter/setter를 통해 initial_point 설정
        self.initial_point = initial_point
        self.callback = callback

    @property
    def initial_point(self) -> np.ndarray | None:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: np.ndarray | None) -> None:
        self._initial_point = value

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: List[BaseOperator] | None = None # aux_operators 인자 추가
    ) -> VQEResult:
        """
        주어진 해밀토니안 연산자의 최소 고유값을 계산합니다.
        Args:
            operator: 최소 고유값을 찾을 해밀토니안 연산자.
            aux_operators: (선택 사항) 최적화된 상태에서 평가할 보조 연산자 리스트.
        Returns:
            VQEResult 객체.
        """
        initial_point = self.initial_point

        # FAdamOptimizer는 bounds를 직접 처리하지 않으므로, None을 전달합니다.
        bounds_for_optimizer = None 

        start_time = time()

        # 에너지 평가 함수 생성
        evaluate_energy = self._get_evaluate_energy(self.ansatz, operator)

        # 기울기 평가 함수 생성 (gradient_calculator가 제공된 경우)
        evaluate_gradient = None
        if self.gradient is not None:
            # _get_evaluate_gradient에 사용자 정의 기울기 계산기 인스턴스를 전달합니다.
            evaluate_gradient = self._get_evaluate_gradient(self.ansatz, operator, self.gradient)
        
        # 최적화 수행
        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_energy,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds_for_optimizer,
            )
        else:
            optimizer_result = self.optimizer.minimize(
                fun=evaluate_energy,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds_for_optimizer,
            )
            
        optimizer_time = time() - start_time

        logger.info(
            "Optimization complete in %s seconds.\nFound optimal point %s",
            optimizer_time,
            optimizer_result.x,
        )

        # 보조 연산자 평가 (aux_operators가 제공된 경우)
        aux_operators_evaluated = []
        if aux_operators is not None and optimizer_result.x is not None:
            # EstimatorV2의 run()에 보조 연산자 PUBs 구성
            aux_pubs = [(self.ansatz, op, optimizer_result.x.tolist()) for op in aux_operators]
            try:
                aux_job = self.estimator.run(aux_pubs)
                aux_results = aux_job.result()
                for i, aux_op in enumerate(aux_operators):
                    # EstimatorV2의 결과는 .values에서 스칼라 값을 반환할 수 있습니다.
                    aux_value = float(aux_results.values[i])
                    # aux_operators_evaluated는 (BaseOperator, tuple[float, float]) 형태를 기대
                    aux_operators_evaluated.append((aux_op, (aux_value, 0.0))) # 0.0은 std_dev (여기서는 계산 안함)
            except Exception as exc:
                logger.warning(f"Evaluating auxiliary operators failed: {exc}")
                aux_operators_evaluated = [] # 실패 시 빈 리스트 유지


        # VQEResult 객체 빌드 및 반환
        return self._build_vqe_result(
            self.ansatz,
            optimizer_result,
            optimizer_time,
            aux_operators_evaluated, # 평가된 보조 연산자 결과 전달
        )

    @classmethod
    def supports_aux_operators(cls) -> bool:
        # 이 VQE 구현에서 aux_operators를 지원할 수 있도록 True로 설정
        return True 

    def _get_evaluate_energy(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator,
    ) -> Callable[[np.ndarray], float]: # 반환 타입을 float로 명확히
        """주어진 매개변수에 대해 앤자츠의 에너지를 평가하는 함수 핸들을 반환합니다."""
        
        eval_count = 0 # 이 eval_count는 VQE 콜백에 전달될 때 사용됩니다.

        def evaluate_energy(parameters: np.ndarray) -> float: # 반환 타입을 float로 명확히
            nonlocal eval_count

            # parameters (np.ndarray)를 List[float]로 변환하여 PUB 구성
            pub = (ansatz, operator, parameters.tolist()) 

            try:
                job = self.estimator.run([pub]) # Estimator.run은 PUBs 리스트를 받음
                estimator_result = job.result()
            except Exception as exc:
                raise RuntimeError("The primitive job to evaluate the energy failed!") from exc

            # EstimatorV2의 결과는 .values 속성을 통해 직접 기대값을 반환합니다.
            # 단일 PUB이므로 .values는 스칼라 값 또는 1개짜리 np.ndarray일 것입니다.
            energy = float(estimator_result[0].data.evs) # 결과가 1D 배열일 경우 첫 번째 요소, float로 변환

            # 콜백 함수 호출
            if self.callback is not None:
                metadata = estimator_result.metadata 
                # metadata가 단일 PUB에 대해 dict 형태로 올 수도, list[dict]로 올 수도 있습니다.
                # 안전하게 첫 번째 metadata를 가져오고 없으면 빈 dict
                current_metadata = metadata[0] if isinstance(metadata, list) and metadata else {}
                
                eval_count += 1
                # 콜백 시그니처: (eval_count, parameters, value, metadata)
                self.callback(eval_count, parameters, energy, current_metadata)
            
            return energy

        return evaluate_energy

    def _get_evaluate_gradient(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator,
        gradient_calculator: FiniteDiffEstimatorGradientV2, # 사용자 정의 기울기 계산기 인스턴스
    ) -> Callable[[np.ndarray], np.ndarray]:
        """주어진 매개변수에 대해 앤자츠의 기울기를 평가하는 함수 핸들을 반환합니다."""

        def evaluate_gradient(parameters: np.ndarray) -> np.ndarray:
            # FiniteDiffEstimatorGradientV2 인스턴스 자체를 호출합니다.
            # __call__ 메서드가 호출됩니다.
            
            # VQE가 모든 파라미터에 대한 기울기를 요구하므로,
            # parameters 인자에 ansatz의 모든 파라미터를 전달합니다.
            
            # FiniteDiffEstimatorGradientV2의 __call__ 메서드는
            # 단일 np.ndarray 기울기만 반환하도록 구현되었으므로, 해당 반환 값을 그대로 사용합니다.
            gradients = gradient_calculator(
                circuits=[ansatz], 
                observables=[operator], 
                parameter_values=[parameters.tolist()], # parameters를 list로 변환하여 전달
                parameters=[list(ansatz.parameters)] # QuantumCircuit.parameters는 Parameter 객체 리스트
            )
            
            return gradients # 단일 np.ndarray (기울기 벡터)

        return evaluate_gradient

    def _build_vqe_result(
        self,
        ansatz: QuantumCircuit,
        optimizer_result: OptimizerResult,
        optimizer_time: float,
        # aux_operators_evaluated는 이제 compute_minimum_eigenvalue에서 직접 평가되어 전달됩니다.
        aux_operators_evaluated: List[tuple[BaseOperator, tuple[float, float]]], # 실제 반환될 타입에 맞춰 수정
    ) -> VQEResult:
        result = VQEResult()
        result.optimal_circuit = ansatz.copy()
        result.eigenvalue = optimizer_result.fun
        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x 
        result.optimal_parameters = dict(
            zip(self.ansatz.parameters, optimizer_result.x) 
        )
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.aux_operators_evaluated = aux_operators_evaluated # 전달받은 값 할당
        result.optimizer_result = optimizer_result
        return result