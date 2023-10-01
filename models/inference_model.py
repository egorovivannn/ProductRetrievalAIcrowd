from pytorch_lightning import LightningModule
import timm
from torch import nn
import torch
from misc import *

class Net(LightningModule):
    def __init__(self, cfg, embedding_size):
        super(Net, self).__init__()
        
        self.backbone = timm.create_model(
            cfg.backbone, 
            pretrained=False, 
            num_classes=0, 
            global_pool="", 
            in_chans=3, 
            features_only = True
        )

        
        # self.backbone.conv_stem.stride = (2, 2)
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']
        
        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g

        self.global_pool = GeM(p_trainable=True)

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = embedding_size

        self.neck = nn.Sequential(
                nn.Linear(fusion_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, [3, 6, 9])
        self.conv_g = nn.Conv2d(backbone_out,feature_dim_l_g,kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g =  nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()
        
        
    def forward(self, batch):

        x = batch
        
        dev = x.device

        x = self.backbone(x)
        
        x_l = x[-2].permute(0, 3, 1, 2)
        x_g = x[-1].permute(0, 3, 1, 2)
        
        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)
        
        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)
        
        x_g = self.global_pool(x_g)
        x_g = x_g[:,:,0,0]
        
        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:,:,0,0]        
        
        x_emb = self.neck(x_fused)

        return x_emb
