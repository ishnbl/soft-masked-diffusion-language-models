#
# Copyright 2026- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import lightning.pytorch as pl


class FreezeBackboneCallback(pl.Callback):
    """
    Freezes every parameter NOT under `tran_head.*` for the first
    `freeze_until_step` optimizer steps, then unfreezes.

    Why: when finetuning a pretrained backbone with the soft-masking head,
    both the head's lambda schedule and the backbone weights move at step 0.
    The backbone sees a noisy soft-input distribution while it is also being
    updated, which makes lambda's gradient signal noisy and can drive lambda
    toward 0. Freezing the backbone for an initial window lets the head
    settle against a stationary signal; we then unfreeze and finetune jointly.

    Notes
    -----
    * Autograd still propagates gradients THROUGH the frozen backbone to
      reach `tran_head` parameters. `requires_grad=False` only suppresses
      `.grad` accumulation on the frozen leaves, not the JVPs along the path.
    * The optimizer was built with backbone params in its groups; during the
      freeze phase their grads are simply None / zero, so AdamW does not
      update them. Unfreezing is just toggling `requires_grad` back on.
    * Pair with `strategy.find_unused_parameters=True` under DDP (the existing
      script already sets this) so DDP tolerates frozen-leaf params.
    """

    def __init__(self, freeze_until_step: int = 1000):
        super().__init__()
        self.freeze_until_step = int(freeze_until_step)
        self._frozen = False
        self._unfrozen = False

    def _set_backbone_grad(self, pl_module, requires_grad: bool):
        n = 0
        for name, param in pl_module.named_parameters():
            if not name.startswith("tran_head."):
                param.requires_grad = requires_grad
                n += 1
        return n

    def on_train_start(self, trainer, pl_module):
        if self.freeze_until_step <= 0:
            return
        n = self._set_backbone_grad(pl_module, requires_grad=False)
        self._frozen = True
        pl_module.print(
            f"[FreezeBackboneCallback] Froze {n} backbone params "
            f"for the first {self.freeze_until_step} optimizer steps."
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (self._frozen
                and not self._unfrozen
                and trainer.global_step >= self.freeze_until_step):
            n = self._set_backbone_grad(pl_module, requires_grad=True)
            self._unfrozen = True
            pl_module.print(
                f"[FreezeBackboneCallback] Unfroze {n} backbone params "
                f"at step {trainer.global_step}."
            )
