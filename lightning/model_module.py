import pytorch_lightning as pl
from .parser import find_loss

class LightningModule(pl.LightningModule):
    def __init__(self, ):
        super().__init__()
    
    def load_model(self, path):
        '''
        To do
        '''
        pass
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        
    
    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        pass            


