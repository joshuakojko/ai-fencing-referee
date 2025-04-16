import glob
import json
from pathlib import Path


def load_video_labels():
    """
    Load all fencing dataset's per video label files and combine them
    """
    all_clips = {}
    labels_dir = Path("fencing_dataset/per_vid_labels")
    json_files = glob.glob(str(labels_dir / "*.json"))

    if not json_files:
        print(f"Warning: No JSON files found in {labels_dir}")
        return all_clips

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                all_clips.update(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {json_file}: {str(e)}")
        except Exception as e:
            print(f"Error loading file {json_file}: {str(e)}")

    if not all_clips:
        print("Warning: No valid clip data found in JSON files")
    return all_clips
