
from pytorch_lightning import Trainer
from .utils import trainer_hparams_default

class Trainer(Trainer):
    def __init__(self, hparams):
        self.hparams = {k: v for k, v in hparams.items() if k in trainer_hparams_default.keys()}
        super().__init__(**self.hparams)