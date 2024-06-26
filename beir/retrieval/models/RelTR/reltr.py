import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to sys.path
sys.path.append(current_dir)

from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR
import torch
from torch import Tensor
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class reltr_model:
    def __init__(self, model_checkpoint):
        position_embedding = PositionEmbeddingSine(128, normalize=True)
        backbone = Backbone('resnet50', False, False, False)
        backbone = Joiner(backbone, position_embedding)
        backbone.num_channels = 2048
        transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                          dim_feedforward=2048,
                          num_encoder_layers=6,
                          num_decoder_layers=6,
                          normalize_before=False,
                          return_intermediate_dec=True)
        self.model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51, num_entities=100, num_triplets=200)
        ckpt = torch.load(model_checkpoint, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
    
    def forward(self, image):
        outputs = self.model(image)
        return outputs
        