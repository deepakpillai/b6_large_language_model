import torch
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Tuple
from torch import Tensor

class Lion(Optimizer):
    """
    Lion optimizer (Evolution of Gradient Signs) - https://arxiv.org/abs/2302.06675
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 3e-4)
        beta1: coefficient for computing running averages of gradient (default: 0.95)
        beta2: coefficient for computing running averages of gradient (default: 0.98)
        weight_decay: weight decay coefficient (default: 0.1)
    """
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        beta1: float = 0.95,
        beta2: float = 0.98,
        weight_decay: float = 0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["beta1"], group["beta2"]

                # Update momentum
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                exp_avg.copy_(update)

                # Update weights
                update = update.mul(beta2).add(grad, alpha=1 - beta2)
                update = update.sign()
                p.add_(update, alpha=-group["lr"])

        return loss