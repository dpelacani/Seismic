import torch


def ema_smoothing(scalars, weight):  # Weight between 0 and 1
    if len(scalars) > 0:
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value

        return smoothed
    return scalars


class CosineAnnealingLinearWarmUpLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """
    Like Cosine Annealing but with a linear warm up at the beginning.

    T_warm is number of iterations for warm up
    """
    def __init__(self, optimizer, T_max, T_warm, eta_min=0, last_epoch=-1):
        # assert T_warm < T_max
        self.T_warm = T_warm
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self) -> float:
        if self._step_count < self.T_warm:
            return [((base_lr - self.eta_min) / self.T_warm) * self._step_count + self.eta_min for base_lr in self.base_lrs]

        return super().get_lr()


class CosineAnnealingConstantWarmUpLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """
    Like Cosine Annealing but with a linear warm up at the beginning.

    T_warm is number of iterations for warm up
    """
    def __init__(self, optimizer, T_max, T_warm, warm_fac=0.01, eta_min=0, last_epoch=-1):
        # assert T_warm < T_max
        self.T_warm = T_warm
        self.warm_fac = warm_fac
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self) -> float:
        if self._step_count < self.T_warm:
            return [base_lr * self.warm_fac for base_lr in self.base_lrs]

        return super().get_lr()


def set_optim_lr(optimiser, lr=None, factor=None):
    assert (lr is not None or factor is not None)

    if isinstance(lr, float):
        lr = [lr] * len(optimiser.param_groups)

    for i, pg in enumerate(optimiser.param_groups):
        if lr is None:
            pg["lr"] = factor * pg["lr"]
        else:
            pg["lr"] = lr[i]