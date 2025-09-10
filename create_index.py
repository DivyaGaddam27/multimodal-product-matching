# create_index.py
import torch
import faiss
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

# --- Import necessary classes from other files ---
from dataset import ShopeeDataset
from model import MultiModalModel


def create_index(data_dir, model_path="multi_modal_model.pth", index_path="faiss_index.bin"):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Corrected: Use 'train' mode for the full index
    train_dataset = ShopeeDataset(data_dir=data_dir, mode='train', transform=image_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = MultiModalModel().to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    embeddings = []
    ids = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Generating Embeddings"):
            # Corrected: Use the 'anchor' keys from the batch
            image = batch['anchor_image'].to(device)
            text_input_ids = batch['anchor_text']['input_ids'].squeeze(1).to(device)
            text_attention_mask = batch['anchor_text']['attention_mask'].squeeze(1).to(device)

            embedding = model(image, text_input_ids, text_attention_mask)
            embeddings.append(embedding.cpu().numpy())

            ids.extend(batch['posting_id'])

    embeddings = np.vstack(embeddings)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    np.save('posting_ids.npy', np.array(ids))

    print(f"FAISS index saved to {index_path}")
    print("Posting IDs saved.")


if __name__ == "__main__":
    DATA_DIR = "static/shopee-product-matching"
    create_index(DATA_DIR)