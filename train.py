# from models.dolg_efficientnet import Net
from models.clip import Net
# from models.hybrid_transformer import Net
import pandas as pd
import numpy as np
from dataset.dataset import Products_dataset
import pytorch_lightning as pl
import argparse
import importlib
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from MCS2023_baseline.data_utils.dataset import SubmissionDataset

if __name__=='__main__':

    # create a new argument parser
    parser = argparse.ArgumentParser(description="Simple argument parser")
    parser.add_argument("--config", action="store", dest="config_file")
    result = parser.parse_args()
    cfg = importlib.import_module(result.config_file).cfg

    # margins for adaptive_arcface loss
    df = pd.read_csv(cfg.train_csv_path)
    tmp = np.sqrt(1 / np.sqrt(df['class'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * cfg.arcface_m_x + cfg.arcface_m_y


    train_transform = cfg.train_aug
    val_transform = cfg.val_aug

    train_dataset = Products_dataset(
        cfg.df,
        cfg.data_path, mode='train', 
        img_dir='train',
        transform=train_transform,
    )
    val_dataset = Products_dataset(
        cfg.data_path, mode='val', 
        transform=val_transform, 
    )

    # gallery_dataset = SubmissionDataset(
    #     root=cfg.test_dataset_path, annotation_file=cfg.gallery_csv_path,
    #     transforms=cfg.val_aug
    # )

    # query_dataset = SubmissionDataset(
    #     root=cfg.test_dataset_path, annotation_file=cfg.queries_csv_path,
    #     transforms=cfg.val_aug, with_bbox=True
    # )

    # val_dataset = [gallery_dataset, query_dataset]

    model = Net(cfg, margins, train_dataset, val_dataset)
    if cfg.ckpt:
        model = model.load_from_checkpoint(
            checkpoint_path=cfg.ckpt,
            strict=False,
            cfg=cfg,
            margins=margins,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
    callbacks = [
        ModelCheckpoint(
            auto_insert_metric_name=True, 
            save_top_k=3, mode='max', 
            monitor='mAP', verbose=True
            ),
        EarlyStopping(
            monitor="mAP", 
            patience=10, verbose=True, 
            mode="max"
            ),
        LearningRateMonitor(logging_interval='step')
        ]
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1, 
        precision=16, 
        val_check_interval=cfg.eval_every if cfg.eval_every!=1 else None,
        #  detect_anomaly=True,
        num_sanity_val_steps=10,
        accumulate_grad_batches=cfg.grad_accum,
        callbacks=callbacks
                        )

    trainer.fit(
        model, 
        # ckpt_path='/media/ivan/Data1/AirCrowd_Products/src/lightning_logs/version_19/checkpoints/epoch=4-step=12586.ckpt'
    )