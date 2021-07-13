import pytorch_lightning as pl
import numpy as np
import os

class LightningModule(pl.LightningModule):
    def __init__(self, ):
        super().__init__()
        self.train_loss = {}
        self.val_loss = {}
        self.current_fold = None
        self.id_ = None
        
    def load_model(self, path):
        '''
        To do
        '''
        pass
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        
        # log hparams도 알아보기
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.save_neptune('loss/train_step', loss)

        return {'loss': loss, 'pred': pred, 'label': y}
        
    def training_epoch_end(self, training_step_outputs):
        total_loss = np.mean(list(map(lambda x: x['loss'].item(), training_step_outputs)))
        self.train_loss['%s epoch' % self.current_epoch] = total_loss.item()
        self.save_neptune('loss/train_epoch', self.train_loss['%s epoch' % self.current_epoch])

        self.training_epoch_end_adding(training_step_outputs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        
        if len(self.train_loss) != 0:
            self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
            self.save_neptune('loss/valid_step', loss)

        return {'loss': loss, 'pred': pred, 'y': y}


    def validation_epoch_end(self, validation_step_outputs):
        if len(self.train_loss) == 0:
                # sanity check
            pass
        
        else:
            # not sanity check
            total_loss = np.mean(list(map(lambda x: x['loss'].item(), validation_step_outputs)))
            self.val_loss['%s epoch' % self.current_epoch] = total_loss.item()
            self.save_neptune('loss/valid_epoch', self.val_loss['%s epoch' % self.current_epoch])

            if len(self.val_loss) > 1:
                print('\r%s epoch, train_loss : %.4f, val_loss : %.4f -> %.4f\n' % (self.current_epoch, 
                                                                            self.train_loss[list(self.train_loss.keys())[-1]], 
                                                                            self.val_loss[list(self.val_loss.keys())[-2]], 
                                                                            self.val_loss[list(self.val_loss.keys())[-1]]), )
                
                '''trainer에서 받아오기'''
                    
            else:
                print('%s epoch, train_loss : %.4f, val_loss : %.4f' % (self.current_epoch, 
                                                                        self.train_loss[list(self.train_loss.keys())[-1]], 
                                                                        self.val_loss[list(self.val_loss.keys())[-1]]))
                print()
        self.validation_epoch_end_adding(validation_step_outputs)
    
    def save_neptune(self, neptune_folder, values):
        if self.run is not None:
            if self.current_fold is None:
                self.run[neptune_folder].log(values)
            else:
                folder_split = neptune_folder.split('/')
                final_folder = folder_split[-1]
                new_neptune_folder = os.path.join(('/').join(folder_split[:-1]), 
                                                  '%s_fold'%self.current_fold, final_folder)
                self.run[new_neptune_folder].log(values)
        
    def training_epoch_end_adding(self, training_step_outputs):
        '''
        example:
        from neptune.new.types import File

        if self.current_epoch % 10 ==0:
            print('print message per 10 epochs')
            self.run["train/image"].log(File('outputs/image.png'))
        '''
        pass
    
    def validation_epoch_end_adding(self, validation_step_outputs):
        '''
        example:
        from neptune.new.types import File

        if self.current_epoch % 10 ==0:
            print('print message per 10 epochs')
            self.run["train/image"].log(File('outputs/image.png'))
        '''
        pass


