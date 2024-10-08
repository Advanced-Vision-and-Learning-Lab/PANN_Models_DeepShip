
from __future__ import print_function
from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
import torchmetrics
from Utils.Network_functions import initialize_model
from Utils.pytorch_utils import do_mixup

# This code uses a newer version of numpy while other packages use an older version of numpy
# This is a simple workaround to avoid errors that arise from the deprecation of numpy data types
np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

class LitModel(L.LightningModule):

    def __init__(self, Params, model_name, num_classes, Dataset, pretrained_loaded, run_number):
        super().__init__()
        
        self.learning_rate = Params['lr']
        self.run_number = run_number
        self.num_classes = num_classes
        self.model_ft, self.mel_extractor = initialize_model(
            model_name, 
            use_pretrained=Params['use_pretrained'], 
            feature_extract=Params['feature_extraction'], 
            num_classes=num_classes,
            pretrained_loaded=pretrained_loaded,
            d_sr=Params['sample_rate']
        )


        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        # Extract mel spectrogram if not PANN model
        x = self.mel_extractor(x)
        # features are from the feature layer (backbone)
        features, y_pred = self.model_ft(x)
        return features, y_pred


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        features, y_pred = self(x) 
        
        #Perform mixup
        y_one_hot = F.one_hot(y,num_classes=self.num_classes)

        try:
            #For PANN models
            y_one_hot = do_mixup(y_one_hot,self.model_ft.lambdas)
            
        except:
            #For TIMM models
            y_one_hot = do_mixup(y_one_hot,self.mel_extractor.lambdas)
        
        loss = F.cross_entropy(y_pred, y_one_hot)
        
        
        # Convert soft labels to hard labels for accuracy calculation
        y_hard = torch.argmax(y_one_hot, dim=1)
    
        self.train_acc(y_pred, y_hard)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('loss', loss, on_step=True, on_epoch=True)
        
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        features, y_pred = self(x)
        val_loss = F.cross_entropy(y_pred, y)
        
        
        self.val_acc(y_pred, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
    
        return val_loss
 
 
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        features, y_pred = self(x)
        test_loss = F.cross_entropy(y_pred, y)
        
        self.test_acc(y_pred, y)
    
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
    
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

