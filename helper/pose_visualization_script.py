import os
import subprocess


def main():
    for file in os.listdir("fencing_dataset/pose_data"):
        if file.endswith(".npy"):
            file_name = file.split(".")[0]
            video_path = f"fencing_dataset/video_clips/{file_name}.mp4"
            pose_data_path = f"fencing_dataset/pose_data/{file}"
            output_path = f"fencing_dataset/pose_video_clips/{file_name}.mp4"

            # Create output directory if it doesn't exist
            os.makedirs("fencing_dataset/pose_video_clips", exist_ok=True)

            subprocess.run(
                [
                    "python",
                    "helper/pose_visualization.py",
                    video_path,
                    pose_data_path,
                    "--output",
                    output_path,
                ]
            )
            print(f"Processed {file_name}")


if __name__ == "__main__":
    main()
