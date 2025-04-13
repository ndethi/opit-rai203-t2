import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
import sys
from tqdm import tqdm # Added for progress bar during database loading

# Define the EmbeddingNet class here (copied from the notebook)
class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name='resnet18', embedding_size=128, pretrained=True):
        super(EmbeddingNet, self).__init__()

        # Load the pretrained backbone model
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            backbone_output_size = 512
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            backbone_output_size = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Remove the classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Projection head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection_head(features)

        # Normalize embeddings to unit length (important for cosine distance)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

    def get_embedding(self, x):
        return self.forward(x)

# Load the trained model
@st.cache_resource
def load_model(model_path='pet_metric_learning_resnet18_triplet.pth'):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the trained model is in the correct location.")
        return None, None

    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        model = EmbeddingNet(
            backbone_name=checkpoint.get('backbone_name', 'resnet18'), # Default if not found
            embedding_size=checkpoint.get('embedding_size', 128),     # Default if not found
            pretrained=False # Important: Load weights, don't re-download pretrained
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        class_mapping = checkpoint.get('class_mapping', {'idx_to_class': {}}) # Default if not found

        return model, class_mapping
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Preprocessing transformation
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to compute embedding for an image
def get_embedding(model, image):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)

    return embedding

# Function to find similar pets in the database
def find_similar_pets(query_embedding, database_embeddings, database_images, database_labels, top_k=5):
    if database_embeddings is None or len(database_embeddings) == 0:
        return [], [], []
    # Compute cosine similarity
    similarity = torch.matmul(query_embedding, database_embeddings.T)

    # Get top-k indices
    # Ensure k is not larger than the number of items in the database
    actual_k = min(top_k, len(database_labels))
    if actual_k == 0:
        return [], [], []

    similarity_scores, indices = torch.topk(similarity, k=actual_k)
    indices = indices.squeeze().tolist()
    similarity_scores = similarity_scores.squeeze().tolist()

    # Handle case where k=1 or only one result
    if not isinstance(indices, list):
        indices = [indices]
        similarity_scores = [similarity_scores]

    # Return top-k similar images and their labels
    similar_images = [database_images[i] for i in indices]
    similar_labels = [database_labels[i] for i in indices]
    similarities = similarity_scores

    return similar_images, similar_labels, similarities

# Load database images and compute embeddings
@st.cache_data # Cache the loaded data
def load_database(directory, _model): # Pass model explicitly to ensure cache invalidation if model changes
    transform = get_transform()
    image_files = []
    # Look for images in subdirectories as well (common in datasets)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    images = []
    file_paths = []
    embeddings = []
    labels = []

    # Limit database size for demo performance
    max_db_size = 500
    image_files = image_files[:max_db_size]

    st.write(f"Found {len(image_files)} images. Processing up to {max_db_size}...")
    progress_bar = st.progress(0)

    for i, img_path in enumerate(tqdm(image_files, desc="Loading Database")):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            file_paths.append(img_path)

            # Compute embedding
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                embedding = _model(img_tensor)
            embeddings.append(embedding.squeeze())

            # Extract label (assuming format like 'ClassName_123.jpg')
            label = os.path.basename(img_path).split('_')[0]
            labels.append(label)

        except Exception as e:
            # st.warning(f"Could not process {img_path}: {e}") # Optional: show warnings
            continue
        progress_bar.progress((i + 1) / len(image_files))

    # Stack embeddings into a tensor
    if embeddings:
        embeddings_tensor = torch.stack(embeddings)
    else:
        embeddings_tensor = torch.tensor([])

    progress_bar.empty() # Clear progress bar

    return images, embeddings_tensor, labels, file_paths

# Main Streamlit app
def run_app():
    st.set_page_config(layout="wide")
    st.title("🐾 Pet Breed Similarity Search 🐾")
    st.write("Upload a pet image to find similar-looking pets in our database!")

    # --- Sidebar for Configuration ---
    st.sidebar.header("Configuration")
    model_file = st.sidebar.text_input("Model File Path", "pet_metric_learning_resnet18_triplet.pth")
    database_dir = st.sidebar.text_input("Database Directory Path", "./data/oxford-iiit-pet/images") # Default to common location
    top_k_similar = st.sidebar.slider("Number of Similar Pets to Show", 1, 10, 5)

    # --- Load Model ---
    model, class_mapping = load_model(model_file)
    if model is None:
        st.stop() # Stop execution if model loading failed

    idx_to_class = class_mapping.get('idx_to_class', {})

    # --- Load Database ---
    st.sidebar.header("Database")
    if st.sidebar.button("Load/Reload Database"):
        # Clear cache if reload is requested
        st.cache_data.clear()
        st.cache_resource.clear() # Also clear model cache if needed
        st.experimental_rerun() # Rerun to reload with cleared cache

    if not os.path.isdir(database_dir):
        st.warning(f"Database directory not found: '{database_dir}'. Please provide a valid path in the sidebar.")
        database_images, database_embeddings, database_labels, file_paths = [], torch.tensor([]), [], []
    else:
        with st.spinner(f"Loading database from '{database_dir}'... This might take a while."):
            # Pass model to ensure cache works correctly with the specific model instance
            database_images, database_embeddings, database_labels, file_paths = load_database(database_dir, model)
        if not database_images:
            st.error("No images loaded from the database directory. Check the path and image files.")
        else:
            st.sidebar.success(f"Loaded {len(database_images)} images from database.")

    # --- Main Area: Query and Results ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Query Image")
        uploaded_file = st.file_uploader("Choose a pet image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert('RGB')
            st.image(query_image, caption="Your Query Image", use_column_width=True)

    with col2:
        st.header("Similar Pets Found")
        if uploaded_file is not None and len(database_images) > 0:
            if st.button("Find Similar Pets"):
                with st.spinner("Comparing your pet..."):
                    # Get embedding for query image
                    query_embedding = get_embedding(model, query_image)

                    # Find similar images
                    similar_images, similar_labels, similarities = find_similar_pets(
                        query_embedding, database_embeddings, database_images, database_labels, top_k=top_k_similar
                    )

                    # Display results
                    if not similar_images:
                        st.warning("Could not find any similar pets.")
                    else:
                        num_results = len(similar_images)
                        cols = st.columns(num_results)
                        for i, (img, label, similarity) in enumerate(zip(similar_images, similar_labels, similarities)):
                            with cols[i]:
                                st.image(img, caption=f"{label.replace('_', ' ').title()}", use_column_width=True)
                                st.write(f"Similarity: {similarity:.3f}")
        elif uploaded_file is None:
            st.info("Upload an image to start the search.")
        elif len(database_images) == 0:
            st.warning("Database is empty or not loaded. Please check the database path and click 'Load/Reload Database' in the sidebar.")


if __name__ == "__main__":
    run_app()
