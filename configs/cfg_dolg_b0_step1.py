from default_config import basic_cfg
import albumentations as A
import os

cfg = basic_cfg
cfg.debug = True

# paths
cfg.name = os.path.basename(__file__).split(".")[0]


cfg.n_classes = 9758 # products 10k
# cfg.n_classes = 142 # fashion products
cfg.backbone = 'tf_efficientnet_b0_ns'
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
cfg.train_bs = 128
cfg.val_bs = 128
cfg.lr = 0.0001
cfg.img_size = (224, 224)
cfg.data_path = '/media/ivan/Data1/AirCrowd_Products/' # products 10k
# cfg.data_path = '/media/ivan/Data1/AirCrowd_Products/Fashion_products/' # fashion products
cfg.test_dataset_path = '/media/ivan/Data1/AirCrowd_Products/MCS2023_development_test_data/development_test_data/'
cfg.gallery_csv_path = cfg.test_dataset_path + 'gallery.csv'
cfg.queries_csv_path = cfg.test_dataset_path + 'queries.csv'
cfg.val_aicrowd = True # if val on aicrowd data

image_size = cfg.img_size[0]

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
        A.Normalize(p=1), 
    ])

cfg.val_aug = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(p=1)
    ])

