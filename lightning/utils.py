trainer_hparams_default = {
        'logger': True,
        'checkpoint_callback': True,
        'callbacks': None,
        'default_root_dir': None,
        'gradient_clip_val': 0.0,
        'gradient_clip_algorithm': 'norm',
        'process_position': 0,
        'num_nodes': 1,
        'num_processes': 1,
        'gpus': None,
        'auto_select_gpus': False,
        'tpu_cores': None,
        # 'ipus': None,
        'log_gpu_memory': None,
        'progress_bar_refresh_rate': None,
        'overfit_batches': 0.0,
        'track_grad_norm': -1,
        'check_val_every_n_epoch': 1,
        'fast_dev_run': False,
        'accumulate_grad_batches': 1,
        'max_epochs': None,
        'min_epochs': None,
        'max_steps': None,
        'min_steps': None,
        'max_time': None,
        'limit_train_batches': 1.0,
        'limit_val_batches': 1.0,
        'limit_test_batches': 1.0,
        'limit_predict_batches': 1.0,
        'val_check_interval': 1.0,
        'flush_logs_every_n_steps': 100,
        'log_every_n_steps': 50,
        'accelerator': None,
        'sync_batchnorm': False,
        'precision': 32,
        'weights_summary': 'top',
        'weights_save_path': None,
        'num_sanity_val_steps': 2,
        'truncated_bptt_steps': None,
        'resume_from_checkpoint': None,
        'profiler': None,
        'benchmark': False,
        'deterministic': False,
        'reload_dataloaders_every_epoch': False,
        'auto_lr_find': False,
        'replace_sampler_ddp': True,
        'terminate_on_nan': False,
        'auto_scale_batch_size': False,
        'prepare_data_per_node': True,
        'plugins': None,
        'amp_backend': 'native',
        'amp_level': 'O2',
        'distributed_backend': None,
        'move_metrics_to_cpu': False,
        'multiple_trainloader_mode': 'max_size_cycle',
        'stochastic_weight_avg': False
}

# warning
optimizer_hparams_default = {
    'optimizer': 'Adam',
    
    'Adadelta' : {
        'lr': 1.0, 'rho': 0.9, 'eps': 1e-06, 'weight_decay': 0
    }, 
    
    'Adagrad' : {
        'lr': 0.01, 'lr_decay': 0, 'weight_decay': 0, 'initial_accumulator_value': 0, 'eps': 1e-10
    },
     
    'Adam' : {
        'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False
    },
     
    'AdamW' : {
        'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.01, 'amsgrad': False
    }, 
    
    'SparseAdam' : {
        'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08
    }, 
    
    'Adamax' : {
        'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0
    }, 
    
    'ASGD' : {
        'lr': 0.01, 'lambd': 0.0001, 'alpha': 0.75, 't0': 1000000.0, 'weight_decay': 0
    }, 
    
    'LBFGS' : {
        'lr': 1, 'max_iter': 20, 'max_eval': None, 'tolerance_grad': 1e-07, 'tolerance_change': 1e-09, 
        'history_size': 100, 'line_search_fn': None
    }, 
    
    'RMSprop' : {
        'lr': 0.01, 'alpha': 0.99, 'eps': 1e-08, 'weight_decay': 0, 'momentum': 0, 'centered': False
    },
    
    'Rprop' : {
        'lr': 0.01, 'etas': (0.5, 1.2), 'step_sizes': (1e-06, 50)
    }, 
    
    'SGD' : {
        'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False
    }
}

# warning
scheduler_hparams_default = {
    'scheduler': None,
    'last_epoch': -1,
    'verbose': False,
    
    'LambdaLR': {
        'lr_lambda': None 
    },
    
    'MultiplicativeLR': {
        'lr_lambda': None 
    },
    
    'StepLR': {
        'step_size': None, 'gamma':0.1
    },
    
    'MultiStepLR': {
        'milestones': None, 'gamma': 0.1, 
    },
    
    'ExponentialLR': {
        'gamma': None
    },
    
    'CosineAnnealingLR': {
        'T_max': None, 'eta_min': 0, 
    },
    
    'CosineAnnealingWarmRestarts': {
        'T_0': None, 'T_mult': 1, 'eta_min': 0, 
    }
}

dataloader_hparams_default = {
    'train_batch_size': 1, 
    'train_shuffle': False, 
    'train_sampler': None,
    'train_batch_sampler': None, 
    'train_num_workers': 0, 
    'train_collate_fn': None,
    'train_pin_memory': False, 
    'train_drop_last': False, 
    'train_timeout': 0,
    'train_worker_init_fn': None, 
    'train_prefetch_factor': 2,
    'train_persistent_workers': False,
    'validation_batch_size': -1, 
    'validation_shuffle': False, 
    'validation_sampler': None,
    'validation_batch_sampler': None, 
    'validation_num_workers': 0, 
    'validation_collate_fn': None,
    'validation_pin_memory': False, 
    'validation_drop_last': False, 
    'validation_timeout': 0,
    'validation_worker_init_fn': None, 
    'validation_prefetch_factor': 2,
    'validation_persistent_workers': False,
}

model_hparams_default = {
    'loss' : None
}

hparams_default = {}
hparams_default.update(trainer_hparams_default)
hparams_default.update(optimizer_hparams_default)
hparams_default.update(scheduler_hparams_default)
hparams_default.update(dataloader_hparams_default)
hparams_default.update(model_hparams_default)


if __name__ == '__main__':
    print(hparams_default)