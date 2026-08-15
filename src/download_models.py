import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def download_face_landmarker(save_dir: str = None):
    """Downloads the MediaPipe Face Landmarker model task file if not present."""
    if save_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(base_dir, "models")

    os.makedirs(save_dir, exist_ok=True)
    target_path = os.path.join(save_dir, "face_landmarker.task")

    if not os.path.exists(target_path):
        print(f"Downloading MediaPipe Face Landmarker model to {target_path}...")
        urllib.request.urlretrieve(MODEL_URL, target_path)
        print("Download complete!")
    else:
        print(f"Model file already exists at: {target_path}")

    return target_path


if __name__ == "__main__":
    download_face_landmarker()
