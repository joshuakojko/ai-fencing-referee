import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import os


def draw_landmarks(frame, landmarks, color):
    """
    Draw pose landmarks on a frame

    Args:
        frame: The frame to draw on
        landmarks: Numpy array of shape (33, 3) containing x, y, z coordinates
        color: Color of the landmarks and connections
    """
    h, w = frame.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    pixel_landmarks = np.zeros_like(landmarks[:, :2])
    pixel_landmarks[:, 0] = landmarks[:, 0] * w
    pixel_landmarks[:, 1] = landmarks[:, 1] * h

    # Draw keypoints
    for i, (x, y) in enumerate(pixel_landmarks):
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

    # Connect keypoints (simplified connection lines based on human pose)
    connections = [
        # Torso
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        # Left arm
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        # Right arm
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        # Left leg
        (23, 25),
        (25, 27),
        (27, 29),
        (27, 31),
        # Right leg
        (24, 26),
        (26, 28),
        (28, 30),
        (28, 32),
        # Face
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        # Face to torso
        (0, 9),
        (9, 10),
        (10, 11),
        (10, 12),
    ]

    for connection in connections:
        start_idx, end_idx = connection
        if (landmarks[start_idx] != 0).any() and (landmarks[end_idx] != 0).any():
            start_point = (
                int(pixel_landmarks[start_idx, 0]),
                int(pixel_landmarks[start_idx, 1]),
            )
            end_point = (
                int(pixel_landmarks[end_idx, 0]),
                int(pixel_landmarks[end_idx, 1]),
            )
            cv2.line(frame, start_point, end_point, color, 2)


def visualize_video_with_pose(
    video_path, pose_data_path, output_path=None, display=True
):
    """
    Visualize pose detection on video

    Args:
        video_path (str): Path to the original video
        pose_data_path (str): Path to the saved pose detection data (.npy file)
        output_path (str, optional): Path to save the output video
        display (bool): Whether to display the video while processing
    """
    # Load pose data
    pose_data = np.load(pose_data_path)  # Shape: (num_frames, 2, 33, 3)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if pose data matches video frame count
    if len(pose_data) != total_frames:
        print(
            f"Warning: Pose data frames ({len(pose_data)}) does not match video frames ({total_frames})"
        )

    # Setup output video writer if requested
    writer = None
    if output_path:
        try:
            # Ensure the directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Try with different codecs if needed
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Check if writer was successfully created
            if not writer.isOpened():
                print(
                    f"Warning: Could not create video writer with mp4v codec. Trying avc1..."
                )
                writer.release()
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                print(
                    f"Error: Failed to create video writer for {output_path}. Output will not be saved."
                )
                writer = None
        except Exception as e:
            print(f"Error creating video writer: {str(e)}")
            writer = None

    # Define colors for each person
    fencer1_color = (0, 255, 0)  # Green
    fencer2_color = (0, 0, 255)  # Blue

    frame_idx = 0

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(pose_data):
            break

        # Get poses for current frame
        frame_pose_data = pose_data[frame_idx]

        # Draw landmarks for each detected person
        for person_idx, person_landmarks in enumerate(frame_pose_data):
            if np.any(person_landmarks):  # Check if landmarks exist (not all zeros)
                color = fencer1_color if person_idx == 0 else fencer2_color
                draw_landmarks(frame, person_landmarks, color)

        # Add frame number
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Write to output file if requested
        if writer:
            try:
                writer.write(frame)
            except Exception as e:
                print(f"Error writing frame {frame_idx}: {str(e)}")

        # Display if requested
        if display:
            cv2.imshow("Pose Detection Visualization", frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    # Clean up
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames")
    if output_path and writer:
        print(f"Output saved to {output_path}")
    elif output_path:
        print(f"Failed to save output to {output_path}")


def visualize_with_matplotlib(
    video_path, pose_data_path, output_path=None, display_every=1
):
    """
    Alternative visualization using matplotlib for better quality
    This is useful when you want to create high-quality visualizations

    Args:
        video_path (str): Path to the original video
        pose_data_path (str): Path to the saved pose detection data (.npy file)
        output_path (str, optional): Path to save the output video
        display_every (int): Display every nth frame to speed up processing
    """
    # Load pose data
    pose_data = np.load(pose_data_path)  # Shape: (num_frames, 2, 33, 3)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up matplotlib figure
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

    # Define colors for each person
    fencer_colors = ["green", "blue"]

    # Define connections for visualization
    connections = [
        # Torso
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        # Left arm
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        # Right arm
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        # Left leg
        (23, 25),
        (25, 27),
        (27, 29),
        (27, 31),
        # Right leg
        (24, 26),
        (26, 28),
        (28, 30),
        (28, 32),
        # Face
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        # Face to torso
        (0, 9),
        (9, 10),
        (10, 11),
        (10, 12),
    ]

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(pose_data):
            break

        # Skip frames to speed up visualization
        if frame_idx % display_every != 0:
            frame_idx += 1
            continue

        # Clear previous plot
        ax.clear()

        # Display the video frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)

        # Get poses for current frame
        frame_pose_data = pose_data[frame_idx]

        # Draw landmarks for each detected person
        for person_idx, person_landmarks in enumerate(frame_pose_data):
            if np.any(person_landmarks):  # Check if landmarks exist (not all zeros)
                color = fencer_colors[person_idx % len(fencer_colors)]

                # Convert normalized coordinates to pixel coordinates
                x_coords = person_landmarks[:, 0] * width
                y_coords = person_landmarks[:, 1] * height

                # Plot points
                ax.scatter(x_coords, y_coords, c=color, s=30)

                # Draw connections
                for connection in connections:
                    start_idx, end_idx = connection
                    if (person_landmarks[start_idx] != 0).any() and (
                        person_landmarks[end_idx] != 0
                    ).any():
                        ax.plot(
                            [x_coords[start_idx], x_coords[end_idx]],
                            [y_coords[start_idx], y_coords[end_idx]],
                            c=color,
                            linewidth=2,
                        )

        # Add frame number
        ax.text(
            10,
            30,
            f"Frame: {frame_idx}",
            fontsize=12,
            color="white",
            bbox=dict(facecolor="black", alpha=0.5),
        )

        ax.set_title(f"Pose Detection - Frame {frame_idx}")
        ax.axis("off")
        plt.tight_layout()

        # Save frame if requested
        if output_path:
            try:
                output_dir = Path(output_path)
                output_dir.mkdir(exist_ok=True, parents=True)
                plt.savefig(output_dir / f"frame_{frame_idx:04d}.png", dpi=150)
            except Exception as e:
                print(f"Error saving frame {frame_idx}: {str(e)}")

        plt.pause(0.01)  # Short pause to update the plot

        frame_idx += 1

    # Clean up
    cap.release()
    plt.close()

    print(f"Processed {frame_idx} frames")
    if output_path:
        print(f"Output frames saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pose detection on video")
    parser.add_argument("video_path", type=str, help="Path to the original video")
    parser.add_argument(
        "pose_data_path",
        type=str,
        help="Path to the saved pose detection data (.npy file)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save the output video"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable display window"
    )
    parser.add_argument(
        "--use-matplotlib", action="store_true", help="Use matplotlib for visualization"
    )
    parser.add_argument(
        "--display-every",
        type=int,
        default=1,
        help="Display every nth frame (for matplotlib)",
    )

    args = parser.parse_args()

    if args.use_matplotlib:
        visualize_with_matplotlib(
            args.video_path, args.pose_data_path, args.output, args.display_every
        )
    else:
        visualize_video_with_pose(
            args.video_path, args.pose_data_path, args.output, not args.no_display
        )
