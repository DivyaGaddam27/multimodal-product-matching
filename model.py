# model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(ImageEncoder, self).__init__()
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.model.classifier = nn.Identity()
        self.projection = nn.Linear(768, output_dim)
    def forward(self, x):
        features = self.model(x)
        features = features.view(features.size(0), -1)
        output = self.projection(features)
        return nn.functional.normalize(output, p=2, dim=1)

class TextEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(TextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.projection = nn.Linear(384, output_dim)
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embedding = output.last_hidden_state[:, 0, :]
        output = self.projection(sentence_embedding)
        return nn.functional.normalize(output, p=2, dim=1)

class MultiModalModel(nn.Module):
    def __init__(self, image_output_dim=256, text_output_dim=256, final_output_dim=256):
        super(MultiModalModel, self).__init__()
        self.image_encoder = ImageEncoder(output_dim=image_output_dim)
        self.text_encoder = TextEncoder(output_dim=text_output_dim)
        self.fusion_layer = nn.Linear(image_output_dim + text_output_dim, final_output_dim)
    def forward(self, images, text_input_ids, text_attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        combined_features = torch.cat((image_features, text_features), dim=1)
        final_embedding = self.fusion_layer(combined_features)
        return nn.functional.normalize(final_embedding, p=2, dim=1)