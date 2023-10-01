import timm
import torch
from torch import nn
import numpy as np

from timm.models.vision_transformer_hybrid import HybridEmbed    
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import faiss
from utils.metrics import map_per_set
from torch import nn

from models.misc import *
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import pandas as pd
from mean_average_precision import calculate_map

class Net(LightningModule):
    def __init__(self, cfg, margins, train_dataset, val_dataset):
        super(Net, self).__init__()

        self.cfg = cfg
        self.lr = cfg.lr
        self.batch_size = cfg.train_bs
        self.n_classes = self.cfg.n_classes
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=cfg.pretrained, 
                                          num_classes=0, 
                                          in_chans=self.cfg.in_channels)
        embedder = timm.create_model(cfg.embedder, 
                                          pretrained=cfg.pretrained, 
                                          in_chans=self.cfg.in_channels,features_only=True, out_indices=[1])

        
        self.backbone.patch_embed = HybridEmbed(embedder,img_size=cfg.img_size[0], 
                                              patch_size=1, 
                                              feature_size=self.backbone.patch_embed.grid_size, 
                                              in_chans=3, 
                                              embed_dim=self.backbone.embed_dim)
#         if 'efficientnet' in cfg.backbone:
#             backbone_out = self.backbone.num_features
#         else:
#             backbone_out = self.backbone.feature_info[-1]['num_chs']

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            
        if "xcit_small_24_p16" in cfg.backbone:
            backbone_out = 384
        elif "xcit_medium_24_p16" in cfg.backbone:
            backbone_out = 512
        elif "xcit_small_12_p16" in cfg.backbone:
            backbone_out = 384
        elif "xcit_medium_12_p16" in cfg.backbone:
            backbone_out = 512   
        elif "swin" in cfg.backbone:
            backbone_out = self.backbone.num_features
        elif "vit" in cfg.backbone:
            backbone_out = self.backbone.num_features
        elif "cait" in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = 2048 

        self.embedding_size = cfg.embedding_size

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if cfg.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif cfg.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(backbone_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif cfg.neck == "option-X":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )
            
        elif cfg.neck == "option-S":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size),
                Swish_module()
            )

        if not self.cfg.headless:    
            self.head_in_units = self.embedding_size
            self.head = ArcMarginProduct_subcenter(self.embedding_size, self.n_classes)
        if self.cfg.loss == 'adaptive_arcface':
            self.loss_fn = ArcFaceLossAdaptiveMargin(margins,self.n_classes,cfg.arcface_s)
        elif self.cfg.loss == 'arcface':
            self.loss_fn = ArcFaceLoss(cfg.arcface_s,cfg.arcface_m)
        else:
            pass
        
        if cfg.freeze_backbone_head:
            for name, param in self.named_parameters():
                if not 'patch_embed' in name:
                    param.requires_grad = False
        
        
    def forward(self, batch, return_emb=False):

        x = batch['input']

        x = self.backbone(x)

        x_emb = self.neck(x)

        if self.cfg.headless or return_emb:
            return x_emb
            # return {"target": batch['target'], 'embeddings': x_emb}
        
        logits = self.head(x_emb)
#         loss = self.loss_fn(logits, batch['target'].long(), self.n_classes)
        preds = logits.softmax(1)
        preds_conf, preds_cls = preds.max(1)
        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())
            return {'loss': loss, "target": batch['target'], "preds_conf":preds_conf,'preds_cls':preds_cls}
        else:
            loss = torch.zeros((1),device=x.device)
            target = torch.zeros((1),device=x.device)
            return {'loss': loss, "target": target,"preds_conf": preds_conf,'preds_cls': preds_cls,
                    'embeddings': x_emb
                   }


    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False


    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True
    
    def unfreeze_all(self, freeze=[]):
        for name, child in self.named_children():
            for param in child.parameters():
                param.requires_grad = True
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        # scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=2, T_mult=1, eta_max=0.01,  T_up=10, gamma=0.7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='max', factor=0.7, patience=2, 
            min_lr=0.000001, verbose=False
            )
        # return {
        #     "optimizer": optim,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "mAP",
        #         "interval": "epoch",
        #         "frequency": 1
        #         },
        # }
        return optim

    def training_step(self, batch, batch_idx):
        data, labels = batch
        out = self({'input': data, 'target': labels})
        loss = out['loss']

        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if len(batch)==2:
            data, labels = batch
            WITHOUT_LABELS = False
        else:
            data = batch
            WITHOUT_LABELS = True
        embs = self({'input': data}, return_emb=True)
        # embs = out['embeddings'].detach().cpu()
        # loss = out['loss']
        # self.log("val_loss", 0)
        if WITHOUT_LABELS:
            return embs.tolist()
        else:
            return (embs.tolist(), labels.tolist())
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, 
            batch_size=self.cfg.train_bs, 
            num_workers=16, drop_last=True
            )
    
    def val_dataloader(self):
        
        if type(self.val_dataset)==list and len(self.val_dataset):
            gallery_dataset = self.val_dataset[0]
            query_dataset = self.val_dataset[1]
            gallery_loader = DataLoader(
                gallery_dataset, batch_size=self.cfg.val_bs,
                shuffle=False, pin_memory=True, num_workers=8
            )

            query_loader = torch.utils.data.DataLoader(
                query_dataset, batch_size=self.cfg.val_bs,
                shuffle=False, pin_memory=True, num_workers=8
            )
            return gallery_loader, query_loader
        else:
            return DataLoader(
                self.val_dataset, shuffle=False, 
                batch_size=self.cfg.val_bs, num_workers=8
            )

    def validation_epoch_end(self, val_outs, *args, **kwargs):
        if not self.cfg.val_aicrowd:
            bd, labels = [], []
            for i in val_outs:
                bd.extend(i[0])
                labels.extend(i[1])
            bd = np.array(bd, np.float32)
            labels = np.array(labels)

            index = faiss.IndexFlatL2(self.cfg.embedding_size)
            index.add(bd)
            D, I = index.search(bd, 6)
            global_preds = labels[I[:, 1:]]
            global_labels = labels
            acc_1 = map_per_set(global_labels.tolist(), global_preds.tolist(), k=1) 
            acc_5 = map_per_set(global_labels.tolist(), global_preds.tolist(), k=5)
            print(f'Metric mAP@1 = {acc_1}, mAP@5 = {acc_5}')
            self.log('mAP', acc_1)
            return {'log': {'mAP': acc_1}}
        
        else:
            try:
                gallery_embeddings = np.zeros((len(self.val_dataset[0]), self.embedding_size))
                query_embeddings = np.zeros((len(self.val_dataset[1]), self.embedding_size))

                for i, outputs in enumerate(val_outs[0]):
                    gallery_embeddings[
                        i*self.cfg.val_bs:(i*self.cfg.val_bs + self.cfg.val_bs), :
                    ] = outputs

                for i, outputs in enumerate(val_outs[1]):
                    query_embeddings[
                        i*self.cfg.val_bs:(i*self.cfg.val_bs + self.cfg.val_bs), :
                    ] = outputs


                gallery_embeddings = normalize(gallery_embeddings)
                query_embeddings = normalize(query_embeddings)

                distances = pairwise_distances(query_embeddings, gallery_embeddings)
                class_ranks = np.argsort(distances, axis=1)[:, :1000]

                seller_gt = pd.read_csv(self.cfg.gallery_csv_path)
                gallery_labels = seller_gt['product_id'].values
                user_gt = pd.read_csv(self.cfg.queries_csv_path)
                query_labels = user_gt['product_id'].values

                mAP = calculate_map(class_ranks, query_labels, gallery_labels)
            except:
                mAP = -1
            print(f'Metric mAP = {mAP}')
            self.log('mAP', mAP)

            return {"mAP": mAP}