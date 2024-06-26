## PyTorch dependencies
import torch.nn as nn
import torch
from torchvision import models

## Local external libraries
from Utils.TDNN import TDNN
from Utils.Generate_Spatial_Dims import generate_spatial_dimensions


#from src.models import ASTModel
#audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
import pdb

class HistRes(nn.Module):
    
    def __init__(self,histogram_layer,parallel=True,model_name ='resnet18',
                 add_bn=True, scale=5,pretrained=True, TDNN_feats = 1):
        
        #inherit nn.module
        super(HistRes,self).__init__()
        self.parallel = parallel
        self.add_bn = add_bn
        self.scale = scale
        self.model_name = model_name
        self.bn_norm = None
        self.fc = None
        self.dropout = None
    
        #self.dropout = nn.Dropout(0.5)  

        #pdb.set_trace()
    
    
        #Default to use resnet18, otherwise use Resnet50
        #Defines feature extraction backbone model and redefines linear layer
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
                
        elif model_name == "resnet50_wide":
            self.backbone = models.wide_resnet50_2(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
           
            
        elif model_name == "resnet50_next":
            self.backbone = models.resnext50_32x4d(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
            
        elif model_name == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained,memory_efficient=True)
            self.bn_norm = self.backbone.features.norm5
            self.backbone.features.norm5 = nn.Sequential()
            self.backbone.avgpool = nn.Sequential()
            num_ftrs =  self.backbone.classifier.in_features
            self.fc = self.backbone.classifier
            self.backbone.classifier = torch.nn.Sequential()
            
        elif model_name == "efficientnet":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs =  self.backbone.classifier[-1].in_features
            self.fc = self.backbone.classifier[-1]
            self.backbone.classifier[-1] = torch.nn.Sequential()
            
        elif model_name == "regnet":
            self.backbone = models.regnet_x_400mf(pretrained)
            num_ftrs =  self.backbone.fc.in_features
        
        elif model_name == "TDNN":
            self.backbone = TDNN(in_channels=TDNN_feats)
            num_ftrs = self.backbone.fc.in_features
            self.dropout = self.backbone.dropout
            
     
            
        #     self.add_bn = False
            
        #     if self.add_bn:
        #         if self.bn_norm is None:
        #             self.bn_norm = nn.BatchNorm1d(num_ftrs)
        #         else:
        #             pass
            
            
            
        #     self.ln_norm = None if not add_ln else nn.LayerNorm(768)
            
        #     layer_norm = True
            
        #     if parallel:        
        #         if self.ln_norm:
                    
        #             self.fc = nn.Sequential(nn.LayerNorm(num_ftrs), nn.Linear(num_ftrs, 4))
        #         else:
        #             self.fc = nn.Linear(num_ftrs, 4) 



        elif model_name == "AST":
            
            
            if parallel:
                self.backbone = ASTModel(label_dim=768, input_fdim=64, input_tdim=48,
                                         imagenet_pretrain=True, audioset_pretrain=True)
                num_ftrs = 768
                
                self.fc = nn.Sequential(
                        nn.Linear(num_ftrs, 4)
                )
                          
            else:
                self.backbone = ASTModel(label_dim=4, input_fdim=64, input_tdim=48,
                                         imagenet_pretrain=True, audioset_pretrain=True)


            # if self.dropout is None:
            #     self.dropout = nn.Sequential()

            # if self.fc is None:
            #     self.fc = self.backbone.fc
            #     self.backbone.fc = torch.nn.Sequential()

        else:
            raise RuntimeError('{} not implemented'.format(model_name))
            
            
            
            
            
            
        if self.add_bn:
            if self.bn_norm is None:
                self.bn_norm = nn.BatchNorm2d(num_ftrs)
            else:
                pass
        
        
        
        #Add dropout if needed for TDNN models only
        # if self.dropout is None:
        #     self.dropout = nn.Sequential()
            
        
        
        #Define histogram layer and fc
        self.histogram_layer = histogram_layer
        #Change histogram layer pooling to adapt to feature constraint: 
        # number of histogram layer features = number of convolutional features
        output_size = int(num_ftrs / histogram_layer.bin_widths_conv.out_channels)  
        
        
        #output_size = generate_spatial_dimensions(output_size)
        #histogram_layer.hist_pool = nn.AdaptiveAvgPool2d(output_size)



        print('output_size: ', output_size)
        histogram_layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)

        
        if self.fc is None:
            self.fc = self.backbone.fc
            self.backbone.fc = torch.nn.Sequential()
     

    
    def forward(self, x):


        if self.model_name == 'AST':
            x = self.backbone.forward(x)  # Extract features
    

            if self.add_ln:
                x = self.ln_norm(x)
                

            if self.add_bn:
                x = self.bn_norm(x)

            
            if self.parallel:              
                
                batch_size = x.size(0)
     
                
                x = x.view(batch_size, 1, 768)      # x: [128, 1, 768]
                #x = x.view(batch_size, 12, 64)      # x: [128, 12, 64]

                #pdb.set_trace()
                x_hist = self.histogram_layer(x)    # x_hist: [128, 768, 1]
                #x_hist = x_hist.view(batch_size, 1, 768) 

        
                # Flatten x to make it compatible for concatenation with x_hist
                x_flat = torch.flatten(x, start_dim=1)  # Flatten x to 2D 
                x_hist_flat = torch.flatten(x_hist, start_dim=1)  
                

                x_combine = torch.cat((x_flat, x_hist_flat), dim=1)  # [128, 1536]
                
                #x_add = torch.add(x_flat, x_hist_flat)  # [128, 768]
                #output = self.fc(x_add)   
                
                #x_combine = self.dropout(x_combine)
                output = self.fc(x_combine)   # torch.Size([256, 4])
                
            else:
                output = x  
                

    
        # Handling other model architectures
        elif self.model_name in ['resnet18', 'resnet50', 'resnet50_wide', 'resnet50_next', 'densenet121', 'efficientnet', 'regnet', 'TDNN']:
            if self.model_name == 'densenet121':
                x = self.backbone(x).unsqueeze(2).unsqueeze(3)
            
            elif self.model_name == 'efficientnet':
                x = self.backbone.features(x)
                pass
            
            elif self.model_name == 'regnet':
                x = self.backbone.stem(x)
                x = self.backbone.trunk_output(x)
                
            elif self.model_name == 'TDNN':
                x = self.backbone.conv1(x)
                x = self.backbone.nonlinearity(x)
                x = self.backbone.maxpool1(x)
                
                x = self.backbone.conv2(x)
                x = self.backbone.nonlinearity(x)
                x = self.backbone.maxpool2(x)
                
                x = self.backbone.conv3(x)
                x = self.backbone.nonlinearity(x)
                x = self.backbone.maxpool3(x)
                
                x = self.backbone.conv4(x)
                x = self.backbone.nonlinearity(x)
                x = self.backbone.maxpool4(x)
       
                
    
            #All ResNet models
            else:
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)
        
        
            #Pass through histogram layer and pooling layer
            if(self.parallel):
                if self.add_bn:
                    if self.model_name == 'TDNN':
                        x_pool = torch.flatten(x,start_dim=-2)
                        x_pool = self.backbone.conv5(x_pool)
                        x_pool = self.backbone.sigmoid(x_pool)
                        x_pool = self.backbone.avgpool(x_pool)
                        x_pool = torch.flatten(self.bn_norm(x_pool.unsqueeze(-1)),start_dim=1)
                       
                    else:
                        x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
                else:
                    if self.model_name == 'TDNN':
                        x_pool = torch.flatten(x,start_dim=-2)
                        x_pool = self.backbone.conv5(x_pool)
                        x_pool = self.backbone.sigmoid(x_pool)
                        x_pool = self.backbone.avgpool(x_pool)
                        x_pool = torch.flatten(x_pool,start_dim=1)
                    else:
                        x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
      
                x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
                x_combine = torch.cat((x_pool,x_hist),dim=1)
                x_combine = self.dropout(x_combine)
                output = self.fc(x_combine)
            else:
                x = torch.flatten(self.histogram_layer(x),start_dim=1)
                x = self.dropout(x)
                output = self.fc(x)
    
    
        return output

        
