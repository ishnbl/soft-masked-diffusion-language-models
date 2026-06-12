import os
from lightning.pytorch.callbacks import ModelCheckpoint
from huggingface_hub import HfApi


class HFOffloadModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that uploads each saved .ckpt to a HF dataset repo,
    then deletes the local copy (except the most recent `keep_local_n`)."""

    def __init__(self, hf_repo_id, hf_path_prefix="", keep_local_n=2, **kwargs):
        super().__init__(**kwargs)
        self.hf_repo_id = hf_repo_id
        self.hf_path_prefix = hf_path_prefix
        self.keep_local_n = keep_local_n
        self._api = HfApi()
        self._history = []  # local filepaths in save order

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if not trainer.is_global_zero:
            return

        fname = os.path.basename(filepath)
        is_last = (fname == "last.ckpt")

        self._api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=os.path.join(self.hf_path_prefix, fname),
            repo_id=self.hf_repo_id,
            repo_type="dataset",
        )

        if is_last:
            return  # always keep last.ckpt locally for resume

        self._history.append(filepath)
        while len(self._history) > self.keep_local_n:
            old = self._history.pop(0)
            if os.path.exists(old):
                os.remove(old)