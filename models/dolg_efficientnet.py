import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import faiss
import timm
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
                                          global_pool="", 
                                          in_chans=self.cfg.in_channels, features_only = True)

        
        if ("efficientnet" in cfg.backbone) & (self.cfg.stride is not None):
            self.backbone.conv_stem.stride = self.cfg.stride
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']
        
        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = cfg.embedding_size

        self.neck = nn.Sequential(
                nn.Linear(fusion_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )


        if not self.cfg.headless:    
            self.head_in_units = self.embedding_size
            self.head = ArcMarginProduct_subcenter(self.embedding_size, self.n_classes)
        if self.cfg.loss == 'adaptive_arcface':
            self.loss_fn = ArcFaceLossAdaptiveMargin(margins, self.n_classes, cfg.arcface_s)
        elif self.cfg.loss == 'arcface':
            self.loss_fn = ArcFaceLoss(cfg.arcface_s,cfg.arcface_m)
        else:
            pass
        
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, self.cfg.dilations)
        self.conv_g = nn.Conv2d(backbone_out,feature_dim_l_g,kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g =  nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()
#         self.unfreeze_all()
        
        
    def forward(self, batch, return_emb=False):

        x = batch['input']
        
        dev = x.device

        x = self.backbone(x)
        
        if self.cfg.permute_channels:
            x_l = x[-2].permute(0, 3, 1, 2)
            x_g = x[-1].permute(0, 3, 1, 2)
        
        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)
        
        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)
        
        x_g = self.global_pool(x_g)
        x_g = x_g[:, :, 0, 0]
        
        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:, :, 0, 0]        
        
        x_emb = self.neck(x_fused)

        if self.cfg.headless or return_emb:
            return x_emb
            # return {"target": batch['target'], 'embeddings': x_emb}

        
        logits = self.head(x_emb)
        preds = logits.softmax(1)

        preds_conf, preds_cls = preds.max(1)
        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())

            return {'loss': loss, "target": batch['target'], "preds_conf":preds_conf,'preds_cls':preds_cls}
        else:
            loss = torch.zeros((1),device=dev)
            return {'loss': loss, "target": batch['target'],"preds_conf":preds_conf,'preds_cls':preds_cls,
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
            min_lr=0.000001, verbose=True
            )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "mAP",
                "interval": "epoch",
                "frequency": self.cfg.eval_every
                },
        }
        # return optim

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

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
                shuffle=False, num_workers=8
            )

            query_loader = torch.utils.data.DataLoader(
                query_dataset, batch_size=self.cfg.val_bs,
                shuffle=False, num_workers=8
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

                print(gallery_embeddings.shape)
                print(query_embeddings.shape)
                print(distances.shape)

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

