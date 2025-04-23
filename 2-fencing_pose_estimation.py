import logging
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from utils import load_video_labels

logger = logging.getLogger(__name__)

class FencingPoseExtractor:
    def __init__(self):
        self.base_path = Path("fencing_dataset")
        self.clips_dir = self.base_path / "video_clips"
        self.labels_dir = self.base_path / "per_vid_labels"
        self.poses_dir = self.base_path / "pose_data"

        self.clips_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        self.poses_dir.mkdir(exist_ok=True)

    def extract_pose_features(self, video_path):
        """
        Extract pose detection features of two fencers in bout via MediaPipe

        Args:
            video_path (Path): The path to the video clip to extract features from

        Returns:
            np.ndarray: A numpy array of shape (num_frames, 2, 33, 3) containing the pose features of two fencers
        """
        # Initialize MediaPipe pose detection

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = "model/pose_landmarker_heavy.task"

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize storage for landmarks
        all_landmarks = []
        timestamp_ms = 0

        with PoseLandmarker.create_from_options(options) as landmarker:
            # Process frames
            with tqdm(total=frame_count, desc=f"Processing {video_path.name}") as pbar:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    # Convert the frame to RGB for MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=frame_rgb
                    )

                    # Detect pose landmarks
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    timestamp_ms += int(1000 / fps)  # Increment timestamp

                    # Extract landmarks if detected
                    frame_landmarks = []
                    if result.pose_landmarks and len(result.pose_landmarks) == 2:
                        for pose_landmarks in result.pose_landmarks:
                            # Convert landmarks to numpy array
                            landmarks_array = np.array(
                                [[lm.x, lm.y, lm.z] for lm in pose_landmarks]
                            )
                            frame_landmarks.append(landmarks_array)
                        print(frame_landmarks)

                        # TODO: Read prior art on how to handle cases where 2 fencers are not detected
                        # Ensure we always have 2 sets of landmarks per frame
                        # If we don't have exactly 2 poses, use empty arrays of the right shape
                    while len(frame_landmarks) < 2:
                        # Create empty landmarks array with correct shape (33, 3)
                        empty_landmarks = np.zeros((33, 3))
                        frame_landmarks.append(empty_landmarks)

                    # Only keep the first 2 poses if we have more than 2
                    if len(frame_landmarks) > 2:
                        frame_landmarks = frame_landmarks[:2]

                    all_landmarks.append(frame_landmarks)
                    pbar.update(1)

        cap.release()

        # Convert to numpy array with shape (num_frames, 2, 33, 3)
        features = np.array(all_landmarks)
        return features

    def process_all_clips(self):
        """
        Process all video clips and extract pose data
        """
        labels = load_video_labels()

        for video_file in tqdm(list(self.clips_dir.glob("*.mp4"))):
            clip_id = video_file.stem
            if clip_id in labels:
                # Check if features already exist
                feature_file = self.poses_dir / f"{clip_id}.npy"
                if feature_file.exists():
                    print(f"Skipping {clip_id}: Features already extracted")
                    continue

                try:
                    # Extract motion features
                    features = self.extract_pose_features(video_file)

                    # Save features
                    np.save(feature_file, features)
                    print(f"Saved features for {clip_id}")
                except Exception as e:
                    print(f"Error processing {clip_id}: {str(e)}")


# class FencingPriorityDataset(Dataset):
#     def __init__(
#         self,
#         dataset_path="fencing_dataset",
#         split="all",
#         train_ratio=0.8,
#         max_seq_length=200,
#     ):
#         self.base_path = Path(dataset_path)
#         self.poses_dir = self.base_path / "pose_data"
#         self.labels_dir = self.base_path / "per_vid_labels"
#         self.split = split
#         self.train_ratio = train_ratio
#         self.max_seq_length = max_seq_length

#         # Load all labels
#         self.labels = {}
#         for json_file in self.labels_dir.glob("*.json"):
#             with open(json_file, "r") as f:
#                 data = json.load(f)
#                 for clip_id, clip_data in data.items():
#                     self.labels[clip_id] = clip_data

#         print(f"Loaded {len(self.labels)} labels from JSON files")

#         # Get list of available feature files
#         self.all_feature_files = [f for f in self.poses_dir.glob("*_features.npy")]
#         print(f"Found {len(self.all_feature_files)} feature files")

#         # Filter feature files that have corresponding labels
#         self.all_feature_files = [
#             f
#             for f in self.all_feature_files
#             if f.stem.replace("_features", "") in self.labels
#         ]
#         print(
#             f"After filtering, {len(self.all_feature_files)} feature files have matching labels"
#         )

#         # Split data into train and val if needed
#         if split == "all":
#             self.feature_files = self.all_feature_files
#         else:
#             # Shuffle the files
#             random.shuffle(self.all_feature_files)

#             # Split into train and val
#             split_idx = int(len(self.all_feature_files) * train_ratio)

#             if split == "train":
#                 self.feature_files = self.all_feature_files[:split_idx]
#             elif split == "val":
#                 self.feature_files = self.all_feature_files[split_idx:]
#             else:
#                 raise ValueError(f"Invalid split: {split}")

#         print(f"Final dataset size for split '{split}': {len(self.feature_files)}")

#     def __len__(self):
#         return len(self.feature_files)

#     def __getitem__(self, idx):
#         feature_file = self.feature_files[idx]
#         clip_id = feature_file.stem.replace("_features", "")

#         # Load feature data
#         features = np.load(feature_file)

#         # Handle sequence length
#         if len(features) > self.max_seq_length:
#             # If sequence is too long, take a random segment
#             start_idx = random.randint(0, len(features) - self.max_seq_length)
#             features = features[start_idx : start_idx + self.max_seq_length]
#         elif len(features) < self.max_seq_length:
#             # If sequence is too short, pad with zeros
#             pad_length = self.max_seq_length - len(features)
#             features = np.pad(features, ((0, pad_length), (0, 0)), mode="constant")

#         # Get label
#         label_data = self.labels[clip_id]
#         label_map = {"left": 0, "right": 1, "together": 2}
#         label = label_map[label_data["annotations"]["label"]]

#         # Convert to tensor
#         features = torch.FloatTensor(features)
#         label = torch.LongTensor([label])

#         return features, label


# class PriorityDetectionModel(nn.Module):
#     def __init__(
#         self, input_size=192, hidden_size=128
#     ):  # 8x8 grid * 3 features per cell
#         super(PriorityDetectionModel, self).__init__()

#         self.lstm = nn.LSTM(
#             input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 3),  # 3 classes: left, right, together
#         )

#     def forward(self, x):
#         # x shape: (batch, frames, features)
#         batch_size = x.size(0)

#         # Pass through LSTM
#         lstm_out, _ = self.lstm(x)

#         # Use last timestep
#         lstm_out = lstm_out[:, -1, :]

#         # Pass through fully connected layers
#         output = self.fc(lstm_out)

#         return output


# def train_model(model, train_loader, val_loader, num_epochs=50):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     best_val_acc = 0

#     for epoch in range(num_epochs):
#         # Training
#         model.train()
#         train_loss = 0
#         correct = 0
#         total = 0

#         for features, labels in tqdm(
#             train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
#         ):
#             features = features.to(device)
#             labels = labels.to(device).squeeze()

#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#         train_acc = 100.0 * correct / total

#         # Validation
#         model.eval()
#         val_loss = 0
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for features, labels in val_loader:
#                 features = features.to(device)
#                 labels = labels.to(device).squeeze()

#                 outputs = model(features)
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()

#         val_acc = 100.0 * correct / total

#         print(
#             f"Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
#         )

#         # Save best model
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_priority_model.pth")


def main():
    logging.basicConfig(level=logging.INFO)

    # Extract motion features from videos
    pose_extractor = FencingPoseExtractor()
    pose_extractor.process_all_clips()

    # # Create datasets
    # train_dataset = FencingPriorityDataset(split="train")
    # val_dataset = FencingPriorityDataset(split="val")

    # # Check if we have enough samples
    # if len(train_dataset) == 0 or len(val_dataset) == 0:
    #     print(
    #         "Not enough samples for train/val split. Using all data for both training and validation."
    #     )
    #     train_dataset = FencingPriorityDataset(split="all")
    #     val_dataset = FencingPriorityDataset(split="all")

    # print(f"Training samples: {len(train_dataset)}")
    # print(f"Validation samples: {len(val_dataset)}")

    # # Adjust batch size if needed
    # batch_size = min(32, len(train_dataset))
    # if batch_size < 32:
    #     print(f"Adjusting batch size to {batch_size} due to small dataset")

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # # Create and train model
    # model = PriorityDetectionModel()
    # train_model(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
