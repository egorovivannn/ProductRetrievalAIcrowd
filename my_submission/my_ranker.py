import numpy as np
import torch

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm

import sys
sys.path.append('./MCS2023_baseline/')
sys.path.append('./models/')

from data_utils.dataset import SubmissionDataset
from inference_clip import Net
from misc import re_ranking
# from configs.cfg_dolg_swin import cfg

import albumentations as A
val_aug = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), p=1)
    ])


class Ranker:
    def __init__(self, dataset_path, gallery_csv_path, queries_csv_path):
        """
        Initialize your model here
        Inputs:
            dataset_path
            gallery_csv_path
            queries_csv_path
        """
        # Try not to change
        self.dataset_path = dataset_path
        self.gallery_csv_path = gallery_csv_path
        self.queries_csv_path = queries_csv_path
        self.max_predictions = 1000

        # Add your code below

        self.batch_size = 32
        # self.embedding_shape = cfg.embedding_size
        self.embedding_shape = 1024
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = Net()
        model.load_state_dict(torch.load('./my_submission/model.pth'))
        self.model = model

        # model = Net(cfg, cfg.embedding_size)
        # self.model = model.load_from_checkpoint(
        #     checkpoint_path="./my_submission/model.ckpt", 
        #     map_location=self.device,
        #     strict=False,
        #     cfg=cfg,
        #     embedding_size=cfg.embedding_size
        # )
        self.model.to(self.device)
        self.model.half()
        self.model.eval()

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def predict_product_ranks(self):
        """
        This function should return a numpy array of shape `(num_queries, 1000)`. 
        For ach query image your model will need to predict 
        a set of 1000 unique gallery indexes, in order of best match first.

        Outputs:
            class_ranks - A 2D numpy array where the axes correspond to:
                          axis 0 - Batch size
                          axis 1 - An ordered rank list of matched image indexes, most confident prediction first
                            - maximum length of this should be 1000
                            - predictions above this limit will be dropped
                            - duplicates will be dropped such that the lowest index entry is preserved
        """

        gallery_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.gallery_csv_path,
            transforms=val_aug
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=8
        )

        query_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.queries_csv_path,
            transforms=val_aug, with_bbox=True
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=8
        )

        print('Calculating embeddings')
        gallery_embeddings = np.zeros((len(gallery_dataset), self.embedding_shape))
        query_embeddings = np.zeros((len(query_dataset), self.embedding_shape))

        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                images = images.to(self.device).half()
                outputs = self.model(images)
                outputs = outputs.data.cpu().numpy()
                gallery_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
            
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                images = images.to(self.device).half()
                outputs = self.model(images)
                outputs = outputs.data.cpu().numpy()
                query_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
        
        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)
        # np.save('local_gallery_embeddings.npy', gallery_embeddings)
        # np.save('local_query_embeddings.npy', query_embeddings)

        # distances = pairwise_distances(query_embeddings, gallery_embeddings)

        qg_distances = pairwise_distances(query_embeddings, gallery_embeddings)
        qq_distances = pairwise_distances(query_embeddings, query_embeddings)
        gg_distances = pairwise_distances(gallery_embeddings, gallery_embeddings)

        distances = re_ranking(qg_distances, qq_distances, gg_distances, 4, 2, 0.5)
        sorted_distances = np.argsort(distances, axis=1)[:, :1000]

        class_ranks = sorted_distances
        return class_ranks

