import torchvision as tv


def get_train_aug(config):
    if config.dataset.augmentations == 'default':
        train_augs = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop((config.dataset.input_size,
                                             config.dataset.input_size)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
    else:
        raise Exception("Unknonw type of augs: {}".format(
            config.dataset.augmentations
        ))
    return train_augs


def get_val_aug(config):
    if config.dataset.augmentations_valid == 'default':
        val_augs = tv.transforms.Compose([
            tv.transforms.Resize((config.dataset.input_size,
                                  config.dataset.input_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
    else:
        raise Exception("Unknonw type of augs: {}".format(
            config.dataset.augmentations
        ))
    return val_augs
