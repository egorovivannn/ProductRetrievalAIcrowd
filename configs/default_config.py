from types import SimpleNamespace
from copy import deepcopy
import numpy as np

cfg = SimpleNamespace(**{})

cfg.n_classes = 9758
cfg.backbone = 'tf_efficientnet_b5_ns'
cfg.pretrained = True
cfg.in_channels = 3
cfg.stride = (2, 2)
cfg.pool = "gem"
cfg.embedding_size = 512
cfg.headless = False
cfg.loss = 'adaptive_arcface'
cfg.arcface_s = 45
cfg.arcface_m = 0.3
cfg.dilations = [3, 6, 9]
cfg.gem_p_trainable = True
cfg.arcface_m_x =  0.45
cfg.arcface_m_y = 0.05
cfg.device = 'cuda'
cfg.ckpt = None
cfg.train_bs = 8
cfg.val_bs = 8
cfg.lr = 0.0001
cfg.grad_accum = 1
cfg.img_size = (456, 456)
cfg.data_path = '/media/ivan/Data1/AirCrowd_Products/'
cfg.test_dataset_path = '/media/ivan/Data1/AirCrowd_Products/MCS2023_development_test_data/development_test_data/'
cfg.gallery_csv_path = cfg.test_dataset_path + 'gallery.csv'
cfg.queries_csv_path = cfg.test_dataset_path + 'queries.csv'
cfg.val_aicrowd = True # if val on aicrowd data

cfg.permute_channels = False
cfg.eval_every = 1

basic_cfg = cfg