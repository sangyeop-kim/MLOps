
from pytorch_lightning import Trainer

class Trainer(Trainer):
    def __init__(self, hparams):
        super().__init__(**hparams)
    
    # def fit(self, ):
    #     self

    # def test(self,):
    #     pass
    
    def fit_crossvalidation(self, ):
        print('To do')
        pass
    
    def test_crossvalidation(self, ):
        print('To do')
        pass