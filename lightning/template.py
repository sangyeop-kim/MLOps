from .model_module import LightningModule
from .trainer import Trainer
from .utils import *
import types
from .parser import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import copy
from datetime import datetime
import pandas as pd
import os

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
        self.log_folder = 'logs'
        self.hparams.update(hparams_default)
        np.random.seed(self.hparams['random_seed'])
        
    def forward(self, x):
        raise NotImplementedError()
    
    def fit(self, hparams, train_dataset, validation_dataset=None, test_dataset=None):
        '''
        test_dataset은 아직 반영 안함.
        '''
        
        self.__update_hparams(hparams)
        self.__connect_trainer_model()
        self.trainer = Trainer(self.hparams)
        run = self.__run_neptune()
        
        subsetsampler = False
        train_idx = None
        validation_idx = None
        
        if self.hparams['validation_ratio'] is not None:
            train_idx, validation_idx = self.__train_validation_split(range(len(train_dataset)))
            subsetsampler=True
            self.__save_index(train_idx, 'train_idx')
            self.__save_index(validation_idx, 'validation_idx')
            
        train_dataloader = self.__make_dataloader(train_dataset, True, subsetsampler, train_idx)
        validation_dataloader = self.__make_dataloader(train_dataset, False, subsetsampler, validation_idx)
        
        self.trainer.fit(self, train_dataloader, validation_dataloader)

    def test(self, ):
        pass
    
    def fit_crossvalidation(self, hparams, train_dataset, test_dataset = None):
        '''
        test_dataset은 아직 반영 안함.
        GPU 분산처리 가능하게?, multiprocess
        trainer, model 합치는 위치?! 검증 ok
        '''
        
        self.__update_hparams(hparams)
        self.__connect_trainer_model()
        self.trainer = Trainer(self.hparams)
        run = self.__run_neptune()

        
        train_loader_list, validation_loader_list, test_loader_list \
                            = self.__make_crossvalidation_dataloader(train_dataset, test_dataset, 
                                                                     run)
        self.models = [copy.deepcopy(self) for k in range(self.hparams['kfold'])]
        
        for k, (train_dataloader, validation_dataloader, test_dataloader) in \
                    enumerate(zip(train_loader_list, validation_loader_list, test_loader_list)):
            self.trainer.fit(self.models[k], train_dataloader, validation_dataloader)

    
    def test_crossvalidation(self, ):
        pass
    
    
    '''
    complete
    '''
    # def __make_logs_folder(self):
    #     self.id_ = run.fetch()['sys']['id']
    #     os.makedirs('%s/%s' % (self.log_folder, self.id_), exist_ok=True)
    
    def __run_neptune(self):
        try:
            import neptune.new as neptune
        except ModuleNotFoundError:
            pass
        else:
            run = neptune.init(project='%s/%s' % (self.hparams['neptune_workspace'], 
                                                    self.hparams['neptune_project']),
                                    name=self.hparams['name'], source_files=['**/*.py', '*.ipynb'])
            self.id_ = run.fetch()['sys']['id']
            changed = {key: values for key, values in self.hparams.items() if values != hparams_default[key]}
            run['changed_hyperparameter'] = changed
            run['default_hyperparameter'] = {key: values for key, values in hparams_default.items() \
                                                if key not in changed.keys()}
            return run
        
    def __get_path(self, path):
        # make folder if not exist
        path_split = path.replace('\\', '/').split('/')
        os.makedirs(os.path.join(self.log_folder, self.id_, '/'.join(path_split[:-1])), 
                    exist_ok=True)
        
        return os.path.join(self.log_folder, self.id_, path)
    
    def __save_index(self, index, index_nam, run):
        if index is not None and run is not None:
            pd.DataFrame({index_name: index}).to_csv(self.__get_path('index/%s.csv' % index_name), 
                                                                    index=False)
            run['index/%s' % index_name].upload(self.__get_path('index/%s.csv' % index_name))
    
    def __train_validation_split(self, idx):
        train_size = int(len(idx)*(1-self.hparams['validation_ratio']))
        train_idx, validation_idx = train_test_split(idx, train_size=train_size)
        
        return train_idx, validation_idx
    
    def __connect_trainer_model(self):
        self.configure_optimizers = types.MethodType(configure_optimizers, self)
        self.loss = find_loss(self.hparams['loss'])
        if self.loss is None:
            raise Exception('loss is not defined')
        
    def __update_hparams(self, hparams):
        for key, value in hparams.items():
            self.hparams[key] = value
    
    def __make_dataloader(self, dataset, train=True, subsetsampler=False, idx=None):
        batch_size = self.hparams['train_batch_size'] if train else self.hparams['validation_batch_size']
        shuffle = self.hparams['train_shuffle'] if train else self.hparams['validation_shuffle']
        
        if subsetsampler:
            sampler = SubsetRandomSampler(idx)
        else:
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
            
            
    def __make_crossvalidation_dataloader(self, train_dataset, test_dataset, run):
        kfold = KFold(n_splits=self.hparams['kfold'], shuffle=True, 
                      random_state=self.hparams['random_seed'])
        
        train_loader_list = []
        validation_loader_list = []
        test_loader_list = []
        fold_info = {}

        for fold, (train_idx, test_idx) in enumerate(kfold.split(range(len(train_dataset)))):
            
            if test_dataset is not None: # test_dataset 존재
                # self.hparams['validation_ratio'] 무시
                fold_info[fold] = [train_idx, test_idx, None]
                
            else:
                if self.hparams['validation_ratio'] is not None: # validation 지정해줌.
                    train_idx, valid_idx = self.__train_validation_split(train_idx)
                    fold_info[fold] = [train_idx, valid_idx, test_idx]
                else:
                    fold_info[fold] = [train_idx, None, test_idx]

            train_idx, validation_idx, test_idx = fold_info[fold]
            if run is not None:
                self.__save_index(train_idx, '%s_fold/train_idx' % fold, run)
                self.__save_index(validation_idx, '%s_fold/validation_idx' % fold, run)
                self.__save_index(test_idx, '%s_fold/test_idx' % fold, run)
            
            train_loader_list.append(self.__make_dataloader(train_dataset, True, subsetsampler=True, 
                                                       idx=train_idx))
            
            if test_idx is not None:
                test_loader_list.append(self.__make_dataloader(train_dataset, False, subsetsampler=True, 
                                                          idx=test_idx))
                if validation_idx is not None:
                    validation_loader_list.append(self.__make_dataloader(train_dataset, False, 
                                                                    subsetsampler=True, 
                                                                    idx=validation_idx))
                else:
                    validation_loader_list = [None for i in range(self.hparams['kfold'])]
                    
            else:
                test_loader_list.append(self.__make_dataloader(test_dataset, False, subsetsampler=False))
                validation_loader_list = [None for i in range(self.hparams['kfold'])]
            
        return train_loader_list, validation_loader_list, test_loader_list