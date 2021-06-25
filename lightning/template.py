from .model_module import LightningModule
from .trainer import Trainer
from .utils import hparams
import types
from .parser.optimizer_parser import find_optimizer
from .parser.scheduler_parser import find_scheduler
from .parser.loss import find_loss

def configure_optimizers(self):
    self.optimizer = find_optimizer(self, self.hparams)
    self.scheduler = find_scheduler(self.optimizer, self.hparams)
    
    if self.scheduler is not None:
        return [self.optimizer], [self.scheduler]
    else:
        return [self.optimizer]

class Template(LightningModule):
    def __init__(self, ):
        super().__init__()
        self.hparams = hparams
        
    def __connect_trainer_model(self, hparams):
        for key, value in hparams.items():
            self.hparams[key] = value
        self.optimizer = find_optimizer()
        self.scheduler = find_scheduler()
        self.configure_optimizers = types.MethodType(configure_optimizers, self)
    
    def forward(self, x):
        raise NotImplementedError()
    
    def fit(self, hparams, train_dataset, validation_dataset = None):        
        self.__connect_trainer_model(hparams)
        self.trainer = Trainer(self.hparams)
    
    def test(self, ):
        pass