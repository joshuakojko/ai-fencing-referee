# AI Fencing Referee

### Setup
```bash
chmod +x setup.sh
./setup.sh
```

### Fencing video download script
```bash
python download_fencing_clips.py
```

### Extract pose detection features from video dataset
```bash
python fencing_priority_detector.py
```

### Visualize pose detection
Visualizes every clip_id with pose detection features extracted
```bash
python helper/pose_visualization_script.py
```
Visualizes specific clip_id with pose detection features extracted
```bash
python helper/pose_visualization /fencing_dataset/video_clips{clip_id}.mp4 /fencing_dataset/pose_data/{clip_id}.npy --output /fencing_dataset/pose_video_clips{clip_id}.mp4
```

### Dataset directory structure
```
├── fencing_dataset
│   ├── per_vid_labels      // labelled JSON dataset from Sholto Douglas
│   ├── pose_data           // Numpy arrays with pose detection features (clip_id).npy
│   ├── pose_video_clips    // 5s fencing dataset clips with pose detection visualized {clip_id}.mp4
│   └── video_clips         // 5s fencing dataset clips {clip_id}.mp4
```