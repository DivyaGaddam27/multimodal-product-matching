# app.py
from flask import Flask, request, jsonify, render_template, url_for
import torch
import faiss
import numpy as np
import os
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
import io
import re
from collections import defaultdict
import torch.nn as nn
from tqdm import tqdm

# --- Import necessary classes from other files ---
from dataset import ShopeeDataset
from model import MultiModalModel, ImageEncoder, TextEncoder

app = Flask(__name__)

# --- Load the trained model and FAISS index on app startup ---
device = torch.device("cpu")
model = MultiModalModel().to(device)
model.load_state_dict(torch.load("multi_modal_model.pth", map_location=device))
model.eval()

faiss_index = faiss.read_index("faiss_index.bin")
posting_ids = np.load("posting_ids.npy")
full_df = pd.read_csv(os.path.join("static", "shopee-product-matching", "train.csv"))

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(image, text):
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    text_tokens = tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")

    with torch.no_grad():
        embedding = model(image_tensor, text_tokens['input_ids'].to(device), text_tokens['attention_mask'].to(device))

    return embedding.cpu().numpy()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files or not request.form.get('text'):
        return jsonify({'error': 'Image and text are required'}), 400

    image_file = request.files['image']
    text_query = request.form['text']

    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {e}'}), 400

    query_embedding = get_embedding(image, text_query)
    distances, indices = faiss_index.search(query_embedding, k=5)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        posting_id = posting_ids[idx]
        item = full_df[full_df['posting_id'] == posting_id].iloc[0]

        image_path_unix = os.path.join("shopee-product-matching", "train_images", item['image']).replace('\\', '/')

        results.append({
            'posting_id': item['posting_id'],
            'title': item['title'],
            'distance': float(distances[0][i]),
            'image_url': url_for('static', filename=image_path_unix)
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)