# Fencing Priority Detection System

This repository contains a machine learning system for detecting priority (right of way) in fencing matches using computer vision and deep learning techniques. The system analyzes fencing videos to determine which fencer has priority during bouts.

## Overview

The system consists of three main components:

1. **Data Collection and Processing** (`download_fencing_clips.py`)

   - Downloads and processes fencing match videos
   - Extracts relevant clips for analysis
   - Organizes data into a structured dataset

2. **Priority Detection Model** (`fencing_priority_detector.py`)

   - Implements a neural network for priority detection
   - Extracts motion features from video frames
   - Trains on labeled fencing match data
   - Includes the `FencingPoseExtractor` class for feature extraction
   - Contains the `PriorityDetectionModel` neural network architecture

3. **Video Analysis** (`analyze_fencing_video.py`)
   - Processes fencing videos in real-time
   - Visualizes priority predictions
   - Generates analysis reports
   - Supports batch processing of multiple videos

## Requirements

- Python 3.7+
- PyTorch >= 1.8.0
- OpenCV >= 4.5.1
- NumPy >= 1.19.2
- tqdm >= 4.59.0
- pathlib >= 1.0.1

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── fencing_dataset/           # Dataset directory
│   ├── video_clips/          # Processed video clips
│   ├── per_vid_labels/       # Priority labels
│   └── pose_data/            # Extracted pose features
├── analyze_fencing_video.py   # Video analysis script
├── fencing_priority_detector.py # Core model implementation
├── download_fencing_clips.py  # Data collection script
├── best_priority_model.pth    # Trained model weights
└── requirements.txt          # Project dependencies
```

## Usage

1. **Data Collection**

   ```bash
   python download_fencing_clips.py
   ```

2. **Training the Model**

   ```bash
   python fencing_priority_detector.py
   ```

3. **Analyzing Videos**
   ```bash
   python analyze_fencing_video.py --video path/to/video.mp4
   ```

## Model Architecture

The priority detection model uses a neural network that:

- Processes motion features extracted from video frames
- Analyzes temporal patterns in fencer movements
- Outputs priority predictions (left, right, or simultaneous)
- Includes confidence scores for predictions

## Visualization

The system generates visualizations showing:

- Priority arrows indicating which fencer has right of way
- Confidence scores for predictions
- Temporal analysis of priority changes

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Dataset Structure

The system expects the following directory structure:

```
fencing_dataset/
├── video_clips/         # Contains MP4 video clips
├── per_vid_labels/      # Contains JSON label files
└── pose_data/          # Will be created to store extracted motion features
```

## Label Format

Each JSON label file contains annotations in the following format:

```json
{
    "clip_id": {
        "annotations": {
            "label": "right|left|together",
            "segment": [start_time, end_time]
        },
        "subset": "train|val|test"
    }
}
```

## Feature Extraction

The system uses the following techniques to extract motion features:

- Background subtraction to detect moving objects
- Grid-based motion analysis to capture spatial patterns
- Motion intensity and direction calculation for each grid cell

## Output

The model classifies each action into one of three categories:

- `left`: Left fencer has priority
- `right`: Right fencer has priority
- `together`: Simultaneous action (no clear priority)

The training process will display accuracy metrics for both training and validation sets.
