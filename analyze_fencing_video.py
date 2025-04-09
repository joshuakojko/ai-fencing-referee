import cv2
import numpy as np
import torch
import json
import shutil
from pathlib import Path
from fencing_priority_detector import FencingPoseExtractor, PriorityDetectionModel

def create_priority_visualization(frame, prediction, score, frame_width, frame_height):
    """Create a minimal visualization of the priority prediction with just an arrow above scoreboard"""
    # Create color map for predictions
    color_map = {
        0: (0, 0, 255),    # Left priority (red)
        1: (0, 255, 0),    # Right priority (green)
        2: (255, 255, 255) # Together (white)
    }
    
    color = color_map[prediction]
    
    # Calculate arrow position and size - positioned just above the scoreboard
    arrow_y = int(frame_height * 0.92)  # Lower position to be just above scoreboard
    arrow_length = 40  # Shorter, more compact arrows
    arrow_thickness = 4  # Slightly thicker arrows
    arrow_spacing = 15  # More space between arrows
    
    if prediction != 2:  # If there's a clear priority
        if prediction == 0:  # Left priority - multiple arrows pointing left
            for i in range(3):  # Draw 3 arrows
                start_x = frame_width // 2 + (i * arrow_spacing)
                start_point = (start_x + arrow_length, arrow_y)
                end_point = (start_x, arrow_y)
                cv2.arrowedLine(frame, start_point, end_point, color, arrow_thickness, tipLength=0.5)
        else:  # Right priority - multiple arrows pointing right
            for i in range(3):  # Draw 3 arrows
                start_x = frame_width // 2 - (i * arrow_spacing)
                start_point = (start_x - arrow_length, arrow_y)
                end_point = (start_x, arrow_y)
                cv2.arrowedLine(frame, start_point, end_point, color, arrow_thickness, tipLength=0.5)
    
    return frame

def setup_output_directories(base_path):
    """Create organized output directory structure"""
    output_dir = base_path / "analysis_output"
    categories = {
        "correct_predictions": ["left", "right", "together"],
        "incorrect_predictions": ["left", "right", "together"]
    }
    
    # Create main output directory
    output_dir.mkdir(exist_ok=True)
    
    # Create category subdirectories
    for category, subcategories in categories.items():
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        for subcategory in subcategories:
            (category_dir / subcategory).mkdir(exist_ok=True)
    
    return output_dir

def get_ground_truth(video_path, labels_dir):
    """Get ground truth label for a video clip"""
    clip_id = video_path.stem
    
    # Search through all JSON files in labels directory
    for json_file in labels_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            labels = json.load(f)
            if clip_id in labels:
                return labels[clip_id]["annotations"]["label"]
    
    return None

def analyze_video(video_path, model_path, labels_dir, output_dir, window_size=15, stride=5):
    """Analyze a video and predict priority"""
    # Check if files exist
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PriorityDetectionModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    # Initialize feature extractor
    extractor = FencingPoseExtractor()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a unique output directory for this clip
    clip_id = video_path.stem
    temp_output_path = output_dir / "temp" / f"{clip_id}_analyzed.mp4"
    temp_output_path.parent.mkdir(exist_ok=True)
    
    # Get ground truth label
    ground_truth = get_ground_truth(video_path, labels_dir)
    if ground_truth is None:
        print(f"Warning: No ground truth label found for {clip_id}")
        return None
    
    # Create output video writer
    output_path = str(temp_output_path)
    
    # Try different codec options
    try:
        # First try HEVC codec (H.265)
        fourcc = cv2.VideoWriter_fourcc(*'hvc1')
    except:
        try:
            # Fall back to H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            # Last resort: try basic codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create video writer with specific parameters for macOS
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
        isColor=True
    )
    
    if not out.isOpened():
        # If failed, try alternative output format
        output_path = str(video_path).replace('.mp4', '_analyzed.mov')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height),
            isColor=True
        )
        if not out.isOpened():
            raise Exception(f"Error creating output video file: {output_path}")
    
    # Buffer for features and frames
    feature_buffer = []
    frame_buffer = []  # Added frame buffer
    current_prediction = None
    current_confidence = None
    
    print("Processing video frames...")
    frame_count = 0
    
    # First pass to get initial prediction
    while len(feature_buffer) < window_size and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_buffer.append(frame.copy())  # Store frame
        
        # Extract features for the current frame
        fg_mask = extractor.bg_subtractor.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate grid features
        grid_size = 8
        cell_width = width // grid_size
        cell_height = height // grid_size
        cell_features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = fg_mask[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                motion_intensity = np.sum(cell > 0) / (cell_width * cell_height)
                sobelx = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=3)
                direction = np.arctan2(np.sum(sobely), np.sum(sobelx))
                cell_features.extend([motion_intensity, np.cos(direction), np.sin(direction)])
        
        feature_buffer.append(cell_features)
    
    # Get initial prediction
    if len(feature_buffer) >= window_size:
        features = np.array(feature_buffer[-window_size:])
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(features_tensor)
            probabilities = torch.softmax(output, dim=1)
            current_prediction = torch.argmax(output, dim=1).item()
            current_confidence = probabilities[0][current_prediction].item()
    
    # Write initial frames with the first prediction
    for frame in frame_buffer:
        frame_with_viz = create_priority_visualization(
            frame.copy(),
            current_prediction if current_prediction is not None else 2,
            current_confidence if current_confidence is not None else 0.0,
            width,
            height
        )
        out.write(frame_with_viz)
    
    # Reset video capture to continue processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, len(frame_buffer))
    
    # Continue processing remaining frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
        
        # Extract features for the current frame
        fg_mask = extractor.bg_subtractor.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate grid features
        grid_size = 8
        cell_width = width // grid_size
        cell_height = height // grid_size
        cell_features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = fg_mask[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                motion_intensity = np.sum(cell > 0) / (cell_width * cell_height)
                sobelx = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=3)
                direction = np.arctan2(np.sum(sobely), np.sum(sobelx))
                cell_features.extend([motion_intensity, np.cos(direction), np.sin(direction)])
        
        feature_buffer.append(cell_features)
        
        # Make prediction when we have enough frames
        if len(feature_buffer) >= window_size:
            # Convert features to tensor
            features = np.array(feature_buffer[-window_size:])
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(features_tensor)
                probabilities = torch.softmax(output, dim=1)
                current_prediction = torch.argmax(output, dim=1).item()
                current_confidence = probabilities[0][current_prediction].item()
            
            # Remove old features based on stride
            if len(feature_buffer) > window_size:
                feature_buffer = feature_buffer[stride:]
        
        # Create visualization for the current frame
        frame_with_viz = create_priority_visualization(
            frame.copy(),
            current_prediction if current_prediction is not None else 2,
            current_confidence if current_confidence is not None else 0.0,
            width,
            height
        )
        
        # Write frame to output video
        out.write(frame_with_viz)
    
    cap.release()
    out.release()
    
    print(f"Analysis complete. Output saved to: {output_path}")
    
    # After processing, determine final prediction
    # Use the most frequent prediction as the final result
    if len(feature_buffer) > 0:
        final_prediction = max(set(feature_buffer), key=feature_buffer.count)
        prediction_map = {0: "left", 1: "right", 2: "together"}
        predicted_label = prediction_map[final_prediction]
        
        # Determine if prediction was correct
        category = "correct_predictions" if predicted_label == ground_truth else "incorrect_predictions"
        
        # Move video to appropriate category directory
        final_output_path = output_dir / category / ground_truth / f"{clip_id}_analyzed.mp4"
        shutil.move(str(temp_output_path), str(final_output_path))
        
        # Save prediction metadata
        metadata = {
            "clip_id": clip_id,
            "ground_truth": ground_truth,
            "predicted": predicted_label,
            "correct": predicted_label == ground_truth,
            "prediction_counts": {
                "left": feature_buffer.count(0),
                "right": feature_buffer.count(1),
                "together": feature_buffer.count(2)
            }
        }
        
        metadata_path = final_output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return final_output_path
    
    return None

def analyze_dataset(dataset_path, model_path):
    """Analyze all videos in the dataset"""
    base_path = Path(dataset_path)
    clips_dir = base_path / "video_clips"
    labels_dir = base_path / "per_vid_labels"
    
    # Setup output directories
    output_dir = setup_output_directories(base_path)
    
    # Process all video clips
    results = {
        "correct_predictions": {"left": 0, "right": 0, "together": 0},
        "incorrect_predictions": {"left": 0, "right": 0, "together": 0},
        "total_clips": 0
    }
    
    for video_path in clips_dir.glob("*.mp4"):
        print(f"\nAnalyzing video: {video_path.name}")
        
        try:
            output_path = analyze_video(video_path, model_path, labels_dir, output_dir)
            if output_path:
                # Update statistics based on the category and ground truth
                category = "correct_predictions" if "correct_predictions" in str(output_path) else "incorrect_predictions"
                ground_truth = get_ground_truth(video_path, labels_dir)
                results[category][ground_truth] += 1
                results["total_clips"] += 1
        except Exception as e:
            print(f"Error processing {video_path.name}: {str(e)}")
    
    # Save overall results
    results_path = output_dir / "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total clips processed: {results['total_clips']}")
    print("\nCorrect Predictions:")
    for category, count in results["correct_predictions"].items():
        print(f"  {category}: {count}")
    print("\nIncorrect Predictions:")
    for category, count in results["incorrect_predictions"].items():
        print(f"  {category}: {count}")

def main():
    try:
        dataset_path = "fencing_dataset"
        model_path = "best_priority_model.pth"
        
        print("Starting dataset analysis...")
        analyze_dataset(dataset_path, model_path)
        print("\nAnalysis complete! Check the analysis_output directory for results.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 