

trainer_hparams = {
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
        'ipus': None,
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
optimizer_hparams = {
    
}

# warning
scheduler_hparams = {
    
}

dataloader_hparams = {
    'batch_size': 1, 
    'shuffle': False, 
    'sampler': None,
    'batch_sampler': None, 
    'num_workers': 0, 
    'collate_fn': None,
    'pin_memory': False, 
    'drop_last': False, 
    'timeout': 0,
    'worker_init_fn': None, 
    'prefetch_factor': 2,
    'persistent_workers': False
}

hparams = {}
hparams.update(trainer_hparams)
hparams.update(optimizer_hparams)
hparams.update(scheduler_hparams)
hparams.update(dataloader_hparams)

if __name__ == '__main__':
    print(hparams)