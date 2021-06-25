import torch
import warnings
from torch.optim import lr_scheduler
warnings.filterwarnings('ignore')

def find_optimizer(self, hparams):
    optimizer_name = hparams['optimizer']
    
    if optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                         rho=self.hparams[optimizer_name]['rho'], 
                                         eps=self.hparams[optimizer_name]['eps'], 
                                         weight_decay=self.hparams[optimizer_name]['weight_decay'])
        
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                        lr_decay=self.hparams[optimizer_name]['lr_decay'], 
                                        weight_decay=self.hparams[optimizer_name]['weight_decay'], 
                                        initial_accumulator_value=\
                                            self.hparams[optimizer_name]['initial_accumulator_value'], 
                                        eps=self.hparams[optimizer_name]['eps'])
        
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                     betas=self.hparams[optimizer_name]['betas'], 
                                     eps=self.hparams[optimizer_name]['eps'], 
                                     weight_decay=self.hparams[optimizer_name]['weight_decay'], 
                                     amsgrad=self.hparams[optimizer_name]['amsgrad'])
        
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                      betas=self.hparams[optimizer_name]['betas'], 
                                      eps=self.hparams[optimizer_name]['eps'], 
                                      weight_decay=self.hparams[optimizer_name]['weight_decay'], 
                                      amsgrad=self.hparams[optimizer_name]['amsgrad'])
        
    elif optimizer_name == 'SparseAdam':
        optimizer = torch.optim.SparseAdam(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                           betas=self.hparams[optimizer_name]['betas'], 
                                           eps=self.hparams[optimizer_name]['eps'])
        
    elif optimizer_name == 'Adamax':
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                       betas=self.hparams[optimizer_name]['betas'], 
                                       eps=self.hparams[optimizer_name]['eps'], 
                                       weight_decay=self.hparams[optimizer_name]['weight_decay'])
        
    elif optimizer_name == 'ASGD':
        optimizer = torch.optim.ASGD(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                     lambd=self.hparams[optimizer_name]['lambd'], 
                                     alpha=self.hparams[optimizer_name]['alpha'], 
                                     t0=self.hparams[optimizer_name]['t0'], 
                                     weight_decay=self.hparams[optimizer_name]['weight_decay'])
        
    elif optimizer_name == 'LBFGS':
        optimizer = torch.optim.LBFGS(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                      max_iter=self.hparams[optimizer_name]['max_iter'], 
                                      max_eval=self.hparams[optimizer_name]['max_eval'], 
                                      tolerance_grad=self.hparams[optimizer_name]['tolerance_grad'], 
                                      tolerance_change=self.hparams[optimizer_name]['tolerance_change'], 
                                      history_size=self.hparams[optimizer_name]['history_size'], 
                                      line_search_fn=self.hparams[optimizer_name]['line_search_fn'])
        
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                        alpha=self.hparams[optimizer_name]['alpha'], 
                                        eps=self.hparams[optimizer_name]['eps'], 
                                        weight_decay=self.hparams[optimizer_name]['weight_decay'], 
                                        momentum=self.hparams[optimizer_name]['momentum'], 
                                        centered=self.hparams[optimizer_name]['centered'])
        
    elif optimizer_name == 'Rprop':
        optimizer = torch.optim.Rprop(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                      etas=self.hparams[optimizer_name]['etas'], 
                                      step_sizes=self.hparams[optimizer_name]['step_sizes'])
        
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams[optimizer_name]['lr'], 
                                    momentum=self.hparams[optimizer_name]['momentum'], 
                                    dampening=self.hparams[optimizer_name]['dampening'], 
                                    weight_decay=self.hparams[optimizer_name]['weight_decay'], 
                                    nesterov=self.hparams[optimizer_name]['nesterov'])
    
    else:
        optimizer_list = "'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', \
        'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD'"
        raise Exception('optimizer should be in [%s]' % optimizer_list)

    return optimizer


def find_scheduler(self, optimizer, hparams):
    scheduler_name = hparams['scheduler']
    if scheduler_name is None:
        return None
    
    if scheduler_name == 'LambdaLR':
        lr_scheduler.LambdaLR(optimizer, last_epoch=self.hparams['last_epoch'], 
                              verbose=self.hparams['verbose'], 
                              lr_lambda=self.hparams[scheduler_name]['lr_lambda'])
        
    elif scheduler_name == 'MultiplicativeLR':
        lr_scheduler.MultiplicativeLR(optimizer, last_epoch=self.hparams['last_epoch'], 
                              verbose=self.hparams['verbose'], 
                              lr_lambda=self.hparams[scheduler_name]['lr_lambda'])

    elif scheduler_name == 'StepLR':
        lr_scheduler.StepLR(optimizer, last_epoch=self.hparams['last_epoch'], 
                              verbose=self.hparams['verbose'], 
                              step_size=self.hparams[scheduler_name]['step_size'], 
                              gamma=self.hparams[scheduler_name]['gamma'])

    elif scheduler_name == 'MultiStepLR':
        lr_scheduler.MultiStepLR(optimizer, last_epoch=self.hparams['last_epoch'], 
                              verbose=self.hparams['verbose'], 
                              milestones=self.hparams[scheduler_name]['milestones'], 
                              gamma=self.hparams[scheduler_name]['gamma'])

    elif scheduler_name == 'ExponentialLR':
        lr_scheduler.ExponentialLR(optimizer, last_epoch=self.hparams['last_epoch'], 
                              verbose=self.hparams['verbose'], 
                              gamma=self.hparams[scheduler_name]['gamma'])
        
    elif scheduler_name == 'CosineAnnealingLR':
        lr_scheduler.CosineAnnealingLR(optimizer, last_epoch=self.hparams['last_epoch'], 
                              verbose=self.hparams['verbose'], 
                              T_max=self.hparams[scheduler_name]['T_max'], 
                              eta_min=self.hparams[scheduler_name]['eta_min'])
    
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        lr_scheduler.CosineAnnealingWarmRestarts(optimizer, last_epoch=self.hparams['last_epoch'], 
                              verbose=self.hparams['verbose'], 
                              T_0=self.hparams[scheduler_name]['T_0'], 
                              T_mult=self.hparams[scheduler_name]['T_mult'], 
                              eta_min=self.hparams[scheduler_name]['eta_min'])
    
    else:
        scheduler_list = ""
        
    
    
    
def find_loss(loss_name):
    loss_name = loss_name.lower().replace(' ', '')
    
    if loss_name == 'L1':
        loss = torch.nn.L1Loss()
        
    elif loss_name == 'MSE':
        loss = torch.nn.MSELoss()
        
    elif loss_name == 'CrossEntropy':
        loss = torch.nn.CrossEntropyLoss()
        
    elif loss_name == 'CTC':
        loss = torch.nn.CTCLoss()
        
    elif loss_name == 'NLL':    
        loss = torch.nn.NLLLoss()
        
    elif loss_name == 'PoissonNLL':
        loss = torch.nn.PoissonNLLLoss()
        
    elif loss_name == 'KLDiv':
        loss = torch.nn.KLDivLoss()
        
    elif loss_name == 'BCE':
        loss = torch.nn.BCELoss()
        
    elif loss_name == 'BCEWithLogits':
        loss = torch.nn.BCEWithLogitsLoss()
        
    elif loss_name == 'MarginRanking':
        loss = torch.nn.MarginRankingLoss()
        
    elif loss_name == 'HingeEmbedding':
        loss = torch.nn.HingeEmbeddingLoss()
        
    elif loss_name == 'MultiLabelMargin':
        loss = torch.nn.MultiLabelMarginLoss()
        
    elif loss_name == 'SmoothL1':
        loss = torch.nn.SmoothL1Loss()
        
    elif loss_name == 'SoftMargin':
        loss = torch.nn.SoftMarginLoss()
        
    elif loss_name == 'MultiLabelSoftMargin':
        loss = torch.nn.MultiLabelSoftMarginLoss()
        
    elif loss_name == 'CosineEmbedding':
        loss = torch.nn.CosineEmbeddingLoss()'
        
    elif loss_name == 'MultiMargin':
        loss = torch.nn.MultiMarginLoss()'
        
    elif loss_name == 'TripletMargin':
        loss = torch.nn.TripletMarginLoss()'
        
    elif loss_name == 'TripletMarginWithDistance':
        loss = torch.nn.TripletMarginWithDistanceLoss()'
        
    else:
        loss_list = "'L1', 'MSE', 'CrossEntropy', 'ce', 'CTC', 'NLL', \
            'PoissonNLL', 'KLDiv', 'BCE', 'BCEWithLogits', 'MarginRanking', \    'HingeEmbedding', 'MultiLabelMargin', 'SmoothL1', 'SoftMargin', \
            'MultiLabelSoftMargin', 'CosineEmbedding', 'MultiMargin', 'TripletMargin', \
            'TripletMarginWithDistance'"
        raise Exception('loss should be in [%s]' % loss_list)
    
    return loss