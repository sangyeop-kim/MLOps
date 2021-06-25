import pytorch_lightning as pl

class LightningModule(pl.LightningModule):
    def __init__(self, ):
        super().__init__()
    
    def load_model(self, path):
        '''
        To do
        '''
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        pass            


