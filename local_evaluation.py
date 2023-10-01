import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from my_submission.aicrowd_wrapper import AIcrowdWrapper
from mean_average_precision import calculate_map

def check_data(datafolder):
    """
    Checks if the data is downloaded and placed correctly
    """
    dl_text = ("Please download the public data from"
               "\n https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023/dataset_files"
               "\n And unzip it with ==> unzip <zip_name> -d public_dataset")
    if not os.path.exists(datafolder):
        raise NameError(f'No folder named {datafolder} \n {dl_text}')
    
    if not os.path.exists(os.path.join(datafolder, 'gallery.csv')):
        raise NameError(f'gallery.csv not found in {datafolder} \n {dl_text}')
    
    if not os.path.exists(os.path.join(datafolder, 'queries.csv')):
        raise NameError(f'queries.csv not found in {datafolder} \n {dl_text}')

def evaluate(LocalEvalConfig):
    """
    Runs local evaluation for the model
    Final evaluation code is the same as the evaluator
    """
    datafolder = LocalEvalConfig.DATA_FOLDER
    
    check_data(datafolder)

    model = AIcrowdWrapper(dataset_dir=datafolder)

    predictions = model.predict_product_ranks()

    seller_gt = pd.read_csv(os.path.join(datafolder, 'gallery.csv'))
    gallery_labels = seller_gt['product_id'].values
    user_gt = pd.read_csv(os.path.join(datafolder, 'queries.csv'))
    query_labels = user_gt['product_id'].values

    # Evalaute metrics
    print("Evaluation Results")
    results = {"mAP": calculate_map(predictions, query_labels, gallery_labels)}
    print(results)


if __name__ == "__main__":
    # change the local config as needed
    class LocalEvalConfig:
        DATA_FOLDER = '/media/ivan/Data1/AirCrowd_Products/MCS2023_development_test_data/development_test_data/'

    evaluate(LocalEvalConfig)
