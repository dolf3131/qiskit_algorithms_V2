from qiskit_algorithms.optimizers import Optimizer
import numpy as np
from scipy.optimize import OptimizeResult
import torch 

class FAdam(Optimizer): 
    def __init__(
        self, 
        lr: float = 1e-3, 
        weight_decay: float = 0.1, 
        betas: tuple[float, float] = (0.9, 0.999), 
        clip: float = 1.0,
        p: float = 0.5, 
        eps: float = 1e-8, 
        epsilon2: float = 0.01, 
        maximize: bool = False, 
        maxiter: int = 100, 
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.clip = clip
        self.p = p
        self.eps = eps
        self.epsilon2 = epsilon2 
        self.maximize = maximize
        self.maxiter = maxiter

        # Qiskit Optimizer
        self._eval_count = 0
        self._num_parameters = None # 

        self._state = {} 
    
    def get_support_level(self):
        """Returns the support level of the optimizer."""
        return {
            "gradient": True,
            "bounds": False, 
            "initial_point": True 
        }
    
    def _get_param_state(self, param_id, param_shape, param_dtype):
        if param_id not in self._state:
            self._state[param_id] = {
                "step": torch.tensor(0.0, dtype=torch.float32),
                "momentum": torch.zeros(param_shape, dtype=param_dtype),
                "fim": torch.ones(param_shape, dtype=param_dtype),
            }
        return self._state[param_id]

    def minimize(self, fun, x0, jac=None, bounds=None):
        if jac is None:
            raise ValueError("FAdam requires a Jacobian (gradient) function. "
                             "Ensure VQE is configured with a gradient estimator.")

        current_params_np = np.array(x0)
        self._num_parameters = len(x0)

        # initial energy state 
        last_eval_energy = fun(current_params_np)
        self._eval_count += 1

        for iteration in range(self.maxiter):
            energy = fun(current_params_np)
            self._eval_count += 1
            gradients_np = jac(current_params_np)

        
            p_torch = torch.tensor(current_params_np, dtype=torch.float32)
            grad_torch = torch.tensor(gradients_np, dtype=torch.float32)

            # maximize
            if self.maximize:
                grad_torch = -grad_torch # [cite: 593]

            state = self._get_param_state(0, p_torch.shape, p_torch.dtype)

            # FAdam update ------------------------------------------
            state["step"] += 1
            step = state["step"].item()

            momentum = state["momentum"]
            fim = state["fim"]
            grad = grad_torch

            beta1, beta2 = self.betas

            # 6 - beta2 bias correction per Section 3.4.4 
            # (1 - beta2^(t-1)) / (1 - beta2^t)
            curr_beta2_denom = (1 - beta2**step)
            # Avoid division by zero if step is 0 (though step starts at 1 here)
            curr_beta2 = beta2 * (1 - beta2**(step - 1)) / curr_beta2_denom if curr_beta2_denom != 0 else beta2 

            # 7 - update fim [cite: 208]
            fim.mul_(curr_beta2).add_(grad * grad, alpha=1 - curr_beta2)

            # 8 - adaptive epsilon 
            rms_grad = torch.sqrt(torch.mean(grad * grad))
            curr_eps = self.eps * max(1.0, rms_grad / self.epsilon2) 

            # 9 - compute natural gradient 
            # fim**p + curr_eps**(2*p)
            fim_base = fim**self.p + curr_eps**(2*self.p) 
            
            # Divide by a small value to prevent division by zero if fim_base becomes zero
            # This is implicitly handled by curr_eps, but adding a tiny const provides robustness
            grad_nat = grad / (fim_base + 1e-10) # 1e-10 is a small constant to prevent division by zero

            # 10 - clip the natural gradient 
            rms_grad_nat = torch.sqrt(torch.mean(grad_nat**2))
            divisor_clip = max(1.0, rms_grad_nat / self.clip)
            grad_nat = grad_nat / divisor_clip

            # 11 - update momentum (no bias correction for momentum) [cite: 216, 264]
            momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

            # 12 - weight decay (natural gradient style) 
            grad_weights = p_torch / (fim_base + 1e-10) # 1e-10 is a small constant to prevent division by zero

            # 13 - clip weight decay 
            rms_grad_weights = torch.sqrt(torch.mean(grad_weights**2))
            divisor_clip_wd = max(1.0, rms_grad_weights / self.clip)
            grad_weights = grad_weights / divisor_clip_wd

            # 14 - compute full update step 
            full_step = momentum + (self.weight_decay * grad_weights)
            lr_step = self.lr * full_step

            # 15 - update weights 
            p_torch.sub_(lr_step)
            # FAdam update end ------------------------------------------

            current_params_np = p_torch.numpy()
            
            # callback (if you want to use it)
            # if self._callback is not None:
            #     # std_dev = 0.0
            #     self._callback(iteration, current_params_np, energy, 0.0)

            last_eval_energy = energy 

        # OptimizeResult
        result = OptimizeResult()
        result.x = current_params_np 
        result.fun = last_eval_energy 
        result.nfev = self._eval_count
        result.nit = self.maxiter
        return result