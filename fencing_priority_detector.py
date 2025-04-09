import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

class FencingPoseExtractor:
    def __init__(self, dataset_path="fencing_dataset"):
        self.base_path = Path(dataset_path)
        self.clips_dir = self.base_path / "video_clips"
        self.labels_dir = self.base_path / "per_vid_labels"
        self.poses_dir = self.base_path / "pose_data"
        self.poses_dir.mkdir(exist_ok=True)
        
        # Initialize OpenCV's background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)

    def load_labels(self):
        """Load all label files"""
        all_labels = {}
        for json_file in self.labels_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                all_labels.update(json.load(f))
        return all_labels

    def extract_motion_features(self, video_path):
        """Extract motion features from a video clip using basic computer vision techniques"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        motion_features = []
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a grid for motion analysis
        grid_size = 8
        cell_width = width // grid_size
        cell_height = height // grid_size
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Calculate motion features for each grid cell
            cell_features = []
            for i in range(grid_size):
                for j in range(grid_size):
                    # Extract cell region
                    cell = fg_mask[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                    
                    # Calculate motion intensity (percentage of white pixels)
                    motion_intensity = np.sum(cell > 0) / (cell_width * cell_height)
                    
                    # Calculate motion direction (using Sobel)
                    sobelx = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=3)
                    direction = np.arctan2(np.sum(sobely), np.sum(sobelx))
                    
                    # Add features
                    cell_features.extend([motion_intensity, np.cos(direction), np.sin(direction)])
            
            # Add frame features
            motion_features.append(cell_features)
            frames.append(frame)
            
        cap.release()
        return np.array(motion_features)

    def process_all_clips(self):
        """Process all video clips and extract motion features"""
        labels = self.load_labels()
        
        for video_file in tqdm(list(self.clips_dir.glob("*.mp4"))):
            clip_id = video_file.stem
            if clip_id in labels:
                # Extract motion features
                features = self.extract_motion_features(video_file)
                
                # Save features
                feature_file = self.poses_dir / f"{clip_id}_features.npy"
                np.save(feature_file, features)

class FencingPriorityDataset(Dataset):
    def __init__(self, dataset_path="fencing_dataset", split="all", train_ratio=0.8, max_seq_length=200):
        self.base_path = Path(dataset_path)
        self.poses_dir = self.base_path / "pose_data"
        self.labels_dir = self.base_path / "per_vid_labels"
        self.split = split
        self.train_ratio = train_ratio
        self.max_seq_length = max_seq_length
        
        # Load all labels
        self.labels = {}
        for json_file in self.labels_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                for clip_id, clip_data in data.items():
                    self.labels[clip_id] = clip_data
        
        print(f"Loaded {len(self.labels)} labels from JSON files")
        
        # Get list of available feature files
        self.all_feature_files = [f for f in self.poses_dir.glob("*_features.npy")]
        print(f"Found {len(self.all_feature_files)} feature files")
        
        # Filter feature files that have corresponding labels
        self.all_feature_files = [f for f in self.all_feature_files if f.stem.replace('_features', '') in self.labels]
        print(f"After filtering, {len(self.all_feature_files)} feature files have matching labels")
        
        # Split data into train and val if needed
        if split == "all":
            self.feature_files = self.all_feature_files
        else:
            # Shuffle the files
            random.shuffle(self.all_feature_files)
            
            # Split into train and val
            split_idx = int(len(self.all_feature_files) * train_ratio)
            
            if split == "train":
                self.feature_files = self.all_feature_files[:split_idx]
            elif split == "val":
                self.feature_files = self.all_feature_files[split_idx:]
            else:
                raise ValueError(f"Invalid split: {split}")
        
        print(f"Final dataset size for split '{split}': {len(self.feature_files)}")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        clip_id = feature_file.stem.replace('_features', '')
        
        # Load feature data
        features = np.load(feature_file)
        
        # Handle sequence length
        if len(features) > self.max_seq_length:
            # If sequence is too long, take a random segment
            start_idx = random.randint(0, len(features) - self.max_seq_length)
            features = features[start_idx:start_idx + self.max_seq_length]
        elif len(features) < self.max_seq_length:
            # If sequence is too short, pad with zeros
            pad_length = self.max_seq_length - len(features)
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
        
        # Get label
        label_data = self.labels[clip_id]
        label_map = {"left": 0, "right": 1, "together": 2}
        label = label_map[label_data["annotations"]["label"]]
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        
        return features, label

class PriorityDetectionModel(nn.Module):
    def __init__(self, input_size=192, hidden_size=128):  # 8x8 grid * 3 features per cell
        super(PriorityDetectionModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, 
                           num_layers=2, batch_first=True, 
                           bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3)  # 3 classes: left, right, together
        )

    def forward(self, x):
        # x shape: (batch, frames, features)
        batch_size = x.size(0)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc(lstm_out)
        
        return output

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features = features.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).squeeze()
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_priority_model.pth")

def main():
    # Extract motion features from videos
    pose_extractor = FencingPoseExtractor()
    pose_extractor.process_all_clips()
    
    # Create datasets
    train_dataset = FencingPriorityDataset(split="train")
    val_dataset = FencingPriorityDataset(split="val")
    
    # Check if we have enough samples
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Not enough samples for train/val split. Using all data for both training and validation.")
        train_dataset = FencingPriorityDataset(split="all")
        val_dataset = FencingPriorityDataset(split="all")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Adjust batch size if needed
    batch_size = min(32, len(train_dataset))
    if batch_size < 32:
        print(f"Adjusting batch size to {batch_size} due to small dataset")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create and train model
    model = PriorityDetectionModel()
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main() 