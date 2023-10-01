from configs.default_config import basic_cfg
import albumentations as A
import os
import pandas as pd
import numpy as np
import timm

cfg = basic_cfg
cfg.debug = True

# paths
cfg.name = os.path.basename(__file__).split(".")[0]

cfg.backbone = 'swin_large_patch4_window12_384'
cfg.pretrained = True
cfg.in_channels = 3
cfg.stride = (2, 2)
cfg.pool = "gem"
cfg.embedding_size = 1024
cfg.headless = False
cfg.loss = 'adaptive_arcface'
# cfg.loss = 'arcface'
cfg.arcface_s = 45
cfg.arcface_m = 0.3
cfg.dilations = [3, 6, 9]
cfg.gem_p_trainable = True
cfg.arcface_m_x =  0.45
cfg.arcface_m_y = 0.05
cfg.device = 'cuda'
cfg.train_bs = 2
cfg.val_bs = 8
cfg.lr = 0.00001
cfg.grad_accum = 32
cfg.eval_every = 0.25
# cfg.img_size = (456, 456)
# cfg.ckpt = '/media/ivan/Data1/AirCrowd_Products/src/lightning_logs/version_19/checkpoints/epoch=4-step=12586.ckpt'
cfg.data_path = '/media/ivan/Data1/AirCrowd_Products/'
# cfg.data_path = '/media/ivan/Data1/AirCrowd_Products/MCS2023_development_test_data/development_test_data/'
# cfg.data_path = '/media/ivan/HDD1/wb_2/'
cfg.test_dataset_path = '/media/ivan/Data1/AirCrowd_Products/MCS2023_development_test_data/development_test_data/'
cfg.gallery_csv_path = cfg.test_dataset_path + 'gallery.csv'
cfg.queries_csv_path = cfg.test_dataset_path + 'queries.csv'
cfg.val_aicrowd = True # if val on aicrowd data

cfg.datasets2exclude = [
    'inshop', 
    'consumer2shop', 
    # 'sunglasses',
    'mybags',
    # 'eyeglasses',
    'products10k'
    ]

# cfg.train_csv_path = os.path.join(cfg.data_path, 'src', 'train.csv')
cfg.train_csv_path = './train.csv'
try:
    df = pd.read_csv(cfg.train_csv_path)
    df = df[df['split']=='train'].reset_index(drop=True)
    if cfg.datasets2exclude:
        datasets2use = [i for i in df['dataset'].unique() if i not in cfg.datasets2exclude]
        df = df[np.sum([df['dataset']==i for i in datasets2use], axis=0, dtype=bool)]
        new_class = {i:idx for idx, i in enumerate(df['class'].unique())}
        df['class'] = df['class'].map(new_class)

    df = df.reset_index(drop=True)
    cfg.n_classes = len(df['class'].unique())
    cfg.df = df
except Exception as e:
    print(e)
# cfg.n_classes = 1827

model = timm.create_model(cfg.backbone, pretrained=False)
mean, std = model.pretrained_cfg['mean'], model.pretrained_cfg['std']
image_size = model.pretrained_cfg['input_size'][2]

if 'swin' in cfg.backbone:
    cfg.permute_channels = True

cfg.img_size = (image_size, image_size)

cfg.train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
        A.OneOf([
            A.ChannelShuffle(),
            A.ChannelDropout(),
            A.ColorJitter(),
            A.ToGray(),
        ], p=0.65),
        A.Normalize(p=1, mean=mean, std=std), 
    ])

cfg.val_aug = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(p=1, mean=mean, std=std)
    ])
