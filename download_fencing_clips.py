import concurrent.futures
import logging
import subprocess
from pathlib import Path

from tqdm import tqdm
import yt_dlp

from utils import load_video_labels

logger = logging.getLogger(__name__)


class FencingClipDownloader:
    def __init__(self):
        self.base_dir = Path("fencing_dataset")
        self.clips_dir = self.base_dir / "video_clips"
        self.labels_dir = self.base_dir / "per_vid_labels"

        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def download_clip(self, clip_id, clip_data):
        """
        Download a specific segment of a video using direct ffmpeg download
        """
        url = clip_data["url"]

        try:
            start_time = float(clip_data["annotations"]["segment"][0])
            end_time = float(clip_data["annotations"]["segment"][1])
        except (ValueError, TypeError) as e:
            raise Exception(f"Invalid timestamp format: {str(e)}")
        except KeyError as e:
            raise Exception(f"Missing required key in clip data: {str(e)}")

        output_path = self.clips_dir / f"{clip_id}.mp4"

        # Skip if clip already exists
        if output_path.exists():
            return f"Skipped {clip_id}: Already exists"

        try:
            # Get the best format URL using yt-dlp
            with yt_dlp.YoutubeDL(
                {"format": "best[height<=720]", "quiet": True}
            ) as ydl:
                info = ydl.extract_info(url, download=False)
                format_url = info["url"]  # Direct URL to the video

            # TODO: Modify ffmpeg command video bitrate for storage efficiency
            # TODO: Consider parallelizing and batching video download and pose detection to reduce storage needs
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                format_url,  # Input is the direct video URL
                "-ss",
                str(start_time),  # Seek before input for faster seeking
                "-t",
                str(end_time - start_time),  # Duration instead of end time
                "-c:v",
                "libx264",  # Copy video to avoid re-encoding
                "-c:a",
                "aac",  # Copy audio to avoid re-encoding
                "-avoid_negative_ts",
                "make_zero",  # Avoid negative timestamps
                "-y",  # Overwrite without asking
                str(output_path),
            ]

            result = subprocess.run(
                ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Verify that the file exists and has reasonable size
            if (
                not output_path.exists() or output_path.stat().st_size < 10000
            ):  # 10KB minimum
                raise Exception("File was not created or is too small")

            return f"Successfully downloaded {clip_id}"

        except subprocess.CalledProcessError as e:
            error_message = f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}"
            if output_path.exists():
                output_path.unlink()
            raise Exception(f"Download failed: {error_message}")
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise Exception(f"Download failed: {str(e)}")

    def download_all_clips(self, max_workers=None):
        """
        Download all clips using parallel processing
        """
        clips_data = load_video_labels()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_clip, clip_id, clip_data): (
                    clip_id,
                    clip_data,
                )
                for clip_id, clip_data in clips_data.items()
            }

            with tqdm(total=len(clips_data), desc="Downloading clips") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    clip_id, clip_data = futures[future]
                    try:
                        result = future.result()
                        print(f"\n{result}")
                    except Exception as e:
                        error_msg = str(e)
                        print(f"\nError processing {clip_id}: {error_msg}")
                    pbar.update(1)

        print(f"\nProcessing complete:")

    def verify_downloads(self):
        """
        Verify all clips were downloaded successfully
        """
        clips_data = load_video_labels()
        missing_clips = []

        for clip_id in clips_data:
            clip_path = self.clips_dir / f"{clip_id}.mp4"
            if not clip_path.exists():
                missing_clips.append(clip_id)

        if missing_clips:
            print(f"\nMissing {len(missing_clips)} clips:")
            for clip_id in missing_clips:
                print(f"- {clip_id}")
        else:
            print("\nAll clips downloaded successfully!")


def main():
    logging.basicConfig(level=logging.INFO)
    downloader = FencingClipDownloader()

    print("Starting clip download process...")
    downloader.download_all_clips()

    print("\nVerifying downloads...")
    downloader.verify_downloads()


if __name__ == "__main__":
    main()
