from .model_module import LightningModule
from .trainer import Trainer
from .utils import *
import types
from .parser import *
from torch.utils.data import DataLoader


def configure_optimizers(self):
    self.optimizer = find_optimizer(self, self.hparams)
    self.scheduler = find_scheduler(self, self.optimizer, self.hparams)
    
    if self.scheduler is not None:
        return [self.optimizer], [self.scheduler]
    else:
        return [self.optimizer]

class Template(LightningModule):
    def __init__(self, ):
        super().__init__()
        self.hparams.update(hparams_default)
    
    def forward(self, x):
        raise NotImplementedError()
    
    def fit(self, hparams, train_dataset, validation_dataset = None):
        self.__update_hparams(hparams)
        self.__connect_trainer_model()
        self.trainer = Trainer(self.hparams)
        
        train_dataloader = self.__make_dataloader(train_dataset, True)
        validation_dataloader = self.__make_dataloader(validation_dataset, False)
        
        self.trainer.fit(self, train_dataloader, validation_dataloader)
    
    def test(self, ):
        pass
    
    def fit_crossvalidation(self, ):
        pass
    
    def test_crossvalidation(self, ):
        pass
    
    
    '''
    complete
    '''
    
    def __connect_trainer_model(self):
        self.configure_optimizers = types.MethodType(configure_optimizers, self)
    
    def __update_hparams(self, hparams):
        for key, value in hparams.items():
            self.hparams[key] = value
            
    def __make_dataloader(self, dataset, train=True):
        batch_size = self.hparams['train_batch_size'] if train else self.hparams['validation_batch_size']
        shuffle = self.hparams['train_shuffle'] if train else self.hparams['validation_shuffle']
        sampler = self.hparams['train_sampler'] if train else self.hparams['validation_sampler']
        batch_sampler = self.hparams['train_batch_sampler'] if train else self.hparams['validation_batch_sampler']
        num_workers = self.hparams['train_num_workers'] if train else self.hparams['validation_num_workers']
        collate_fn = self.hparams['train_collate_fn'] if train else self.hparams['validation_collate_fn']
        pin_memory = self.hparams['train_pin_memory'] if train else self.hparams['validation_pin_memory']
        drop_last = self.hparams['train_drop_last'] if train else self.hparams['validation_drop_last']
        timeout = self.hparams['train_timeout'] if train else self.hparams['validation_timeout']
        worker_init_fn = self.hparams['train_worker_init_fn'] if train else self.hparams['validation_worker_init_fn']
        prefetch_factor = self.hparams['train_prefetch_factor'] if train else self.hparams['validation_prefetch_factor']
        persistent_workers = self.hparams['train_persistent_workers'] \
                                    if train else self.hparams['validation_persistent_workers']

        if dataset is None:
            return None
        else:
            if batch_size <= 0:
                batch_size = len(dataset)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, 
                              batch_sampler=batch_sampler, num_workers=num_workers, timeout=timeout,
                              collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, 
                              worker_init_fn=worker_init_fn, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)