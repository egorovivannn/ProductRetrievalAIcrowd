# ProductRetrievalAIcrowd
Repo for solving visual-product-recognition-challenge-2023 from AIcrowd
Results:
1) We used open-source datasets such as *Products10K*, *Alibaba-v2*, and the datasets from *DeepFashion* called *Consumer2Shop* and *inShop*
2) We lacked of sunglasses and eyeglasses data. We saw models struggling to differ some glasses classes so we collected data from WEB. It was about 2k instances of glasses with reference photos and customer reviews. [LINK](https://www.kaggle.com/datasets/egorovlvan/glasses-dataset)
3) We started training models with EfficientNets, but then shortly figured out that Clip models outperform them by far.
4) The highest scores were gotten with ConvNextXXLarge and it was about 0.62 mAP@1 locally. it is worth noting that local and public scores correlated a lot, so we could get 5 place easily, but failed to fit in constrains of inference of 30 minutes even in half precision.
5) Some products with different colors had to belong to same class, so we tried different Augs related to color namely ChannelShuffle, ToGray, ChannelDropout, and ColorJitter which gave us 1-2% of score.
6) We used [ReRanking](https://github.com/Wanggcong/Spatial-Temporal-Re-identification/blob/master/re_ranking.py) which always gave us +0.02 in mAP score.
7) Щne thing we learned from the competition that we needed to set a very low (maybe 10e-6 or 10e-7) LearningRate for CLIP backbone and a normal one (3e-4) for ArcFace Head and it works like a charm.
