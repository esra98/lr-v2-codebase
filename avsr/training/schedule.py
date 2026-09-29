"""Learning-rate schedule and weight averaging."""

import copy
import math

import torch


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    """Linear warm-up followed by cosine decay to zero, stepped per update.

    ``hold_steps`` gives, for every parameter group, a number of initial
    updates during which the group is not changed at all. Its warm-up starts
    when the hold ends. This keeps pretrained weights intact while layers
    that start from random values send them meaningless gradients.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, hold_steps=None, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        groups = len(optimizer.param_groups)
        self.hold_steps = [int(h) for h in (hold_steps or [0] * groups)]
        if len(self.hold_steps) != groups:
            raise ValueError("hold_steps needs one value per parameter group")
        super().__init__(optimizer, last_epoch)

    def _scale(self, step, hold):
        if step < hold:
            return 0.0
        ramp = min(1.0, (step - hold + 1) / self.warmup_steps)
        if step < self.warmup_steps:
            decay = 1.0
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return ramp * decay

    def get_lr(self):
        step = self.last_epoch
        return [base_lr * self._scale(step, hold) for base_lr, hold in zip(self.base_lrs, self.hold_steps)]


class EMA(torch.nn.Module):
    """Exponential moving average of model weights.

    Batch-norm statistics and other buffers are copied rather than averaged.
    It is a module so that the average follows the model to the GPU and is
    stored in checkpoints.
    """

    def __init__(self, model, decay):
        super().__init__()
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        # The average is only ever used for evaluation.
        return super().train(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)


def average_checkpoints(paths, key="state_dict"):
    """Average the weights stored in several checkpoints."""
    if not paths:
        raise ValueError("No checkpoints to average")
    avg = None
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)[key]
        if avg is None:
            avg = {k: v.clone().double() if v.is_floating_point() else v.clone() for k, v in state.items()}
            continue
        if state.keys() != avg.keys():
            raise ValueError(f"{path} has different parameters from {paths[0]}")
        for k, v in state.items():
            if v.is_floating_point():
                avg[k] += v.double()
            else:
                avg[k] = v.clone()  # counters such as num_batches_tracked: keep the latest
    return {
        k: (v / len(paths)).to(torch.float32) if v.is_floating_point() else v for k, v in avg.items()
    }
