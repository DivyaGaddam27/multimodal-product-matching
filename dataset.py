# dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from collections import defaultdict
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split

class ShopeeDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name='sentence-transformers/all-MiniLM-L6-v2', mode='train', transform=None):
        self.data_dir = data_dir
        if mode == 'train':
            df_path = os.path.join(data_dir, 'train_split_filtered.csv')
        else:
            df_path = os.path.join(data_dir, 'val_split_filtered.csv')
        self.df = pd.read_csv(df_path)
        self.image_dir = os.path.join(data_dir, 'train_images')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.transform = transform
        self.mode = mode
        self.label_groups = self.df.groupby('label_group')['posting_id'].apply(list)
        self.label_group_map = self.df.set_index('posting_id')['label_group'].to_dict()
        if self.mode == 'train':
            self.label_dict = defaultdict(list)
            for i, row in self.df.iterrows():
                self.label_dict[row['label_group']].append(i)

    def __len__(self):
        return len(self.df)

    def get_image(self, image_id):
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def get_text_tokens(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")

    def __getitem__(self, idx):
        while True:
            try:
                anchor_row = self.df.iloc[idx]
                anchor_image = self.get_image(anchor_row['image'])
                anchor_text_tokens = self.get_text_tokens(anchor_row['title'])

                if self.mode == 'train':
                    label_group = anchor_row['label_group']
                    positive_indices = self.label_dict[label_group]
                    positive_idx = np.random.choice([i for i in positive_indices if i != idx])

                    positive_row = self.df.iloc[positive_idx]
                    positive_image = self.get_image(positive_row['image'])
                    positive_text_tokens = self.get_text_tokens(positive_row['title'])

                    negative_label_group = np.random.choice([g for g in self.label_groups.keys() if g != label_group])
                    negative_posting_id = np.random.choice(self.label_groups[negative_label_group])
                    negative_row = self.df[self.df['posting_id'] == negative_posting_id].iloc[0]

                    negative_image = self.get_image(negative_row['image'])
                    negative_text_tokens = self.get_text_tokens(negative_row['title'])

                    return {
                        'anchor_image': anchor_image, 'anchor_text': anchor_text_tokens,
                        'positive_image': positive_image, 'positive_text': positive_text_tokens,
                        'negative_image': negative_image, 'negative_text': negative_text_tokens,
                        'posting_id': anchor_row['posting_id'] # <-- Corrected: Add posting_id
                    }

                else: 
                    return {
                        'image': anchor_image,
                        'text': anchor_text_tokens,
                        'posting_id': anchor_row['posting_id'],
                        'label_group': anchor_row['label_group']
                    }
            except (OSError, FileNotFoundError, ValueError) as e:
                #print(f"Warning: Skipping corrupted, missing, or singleton item at index {idx}. Error: {e}")
                idx = np.random.randint(0, len(self.df))
