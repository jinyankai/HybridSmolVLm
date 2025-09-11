import math
from torch.optim.lr_scheduler import _LRScheduler


class TrapezoidalLRScheduler(_LRScheduler):
    """
    一个三段式学习率调度器，包含线性预热、常数保持和余弦衰减阶段。
    也被称为“梯形学习率调度器”。

    Args:
        optimizer (Optimizer): 包装的优化器。
        num_warmup_steps (int): 线性预热的总步数。
        num_stable_steps (int): 学习率保持不变的总步数。
        num_training_steps (int): 训练的总步数。
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """

    def __init__(self, optimizer, num_warmup_steps: int, num_stable_steps: int, num_training_steps: int,
                 last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        self.num_stable_steps = num_stable_steps
        self.num_training_steps = num_training_steps

        self.decay_start_step = num_warmup_steps + num_stable_steps
        self.num_decay_steps = num_training_steps - self.decay_start_step

        # 确保衰减步数不为负
        if self.num_decay_steps < 0:
            raise ValueError(
                f"The number of training steps ({num_training_steps}) must be greater than "
                f"the sum of warmup steps ({num_warmup_steps}) and stable steps ({num_stable_steps})."
            )

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        根据当前步数 (self.last_epoch) 计算学习率。
        """
        # self.base_lrs 是优化器初始的学习率 (即峰值学习率)
        # _LRScheduler 基类会自动处理，为每个参数组返回一个学习率
        new_lrs = []
        for base_lr in self.base_lrs:
            current_step = self.last_epoch

            if current_step < self.num_warmup_steps:
                # 阶段一：Warm-up
                lr = base_lr * (current_step / self.num_warmup_steps)
            elif current_step < self.decay_start_step:
                # 阶段二：Stable
                lr = base_lr
            else:
                # 阶段三：Decay
                # 计算在衰减阶段的进度
                decay_progress = (current_step - self.decay_start_step) / self.num_decay_steps
                # 使用余弦退火公式
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                lr = base_lr * cosine_decay

            new_lrs.append(lr)

        return new_lrs