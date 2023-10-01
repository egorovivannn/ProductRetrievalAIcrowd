import torch.nn as nn
import torch.nn.functional as F
import open_clip
 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms('convnext_xxlarge', pretrained=False)
        self.vit_backbone = vit_backbone
        self.dim = vit_backbone.token_embedding.embedding_dim

    def forward(self, x):
        x = self.vit_backbone.encode_image(x)
        return x
