import os
import sys
import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Landmark index groups for facial ROIs in MediaPipe 468/478 Face Mesh
FOREHEAD_LANDMARKS = [10, 67, 109, 151, 337, 297, 69, 299, 338, 108]
LEFT_CHEEK_LANDMARKS = [117, 118, 101, 205, 50, 187, 207, 123, 116]
RIGHT_CHEEK_LANDMARKS = [346, 347, 330, 425, 280, 411, 427, 352, 345]


class FaceLandmarkExtractor:
    """MediaPipe Face Landmarker wrapper for rPPG ROI extraction with OpenCV fallback."""

    def __init__(self, model_path: str = None):
        self.use_opencv_fallback = False
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_landmarker.task")

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"MediaPipe model file not found at: {model_path}")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            print("MediaPipe FaceLandmarker loaded successfully!")
        except Exception as e:
            print(f"[Notice] MediaPipe C-bindings unavailable ({e}). Using OpenCV Face ROI fallback.")
            self.use_opencv_fallback = True
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cascade_path = os.path.join(base_dir, "models", "haarcascade_frontalface_default.xml")
            
            if not os.path.exists(cascade_path):
                import urllib.request
                os.makedirs(os.path.dirname(cascade_path), exist_ok=True)
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, cascade_path)

            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Processes a single BGR frame.
        Returns:
            mean_rgb: np.ndarray of shape (3,) with average [R, G, B] values over facial ROIs,
                      or None if no face is detected.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.use_opencv_fallback:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                return None
            
            fx, fy, fw, fh = faces[0]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Forehead ROI
            fh_y1, fh_y2 = fy + int(0.1 * fh), fy + int(0.3 * fh)
            fh_x1, fh_x2 = fx + int(0.25 * fw), fx + int(0.75 * fw)
            mask[max(0, fh_y1):min(h, fh_y2), max(0, fh_x1):min(w, fh_x2)] = 1
            
            # Left & Right Cheek ROIs
            ck_y1, ck_y2 = fy + int(0.5 * fh), fy + int(0.75 * fh)
            lc_x1, lc_x2 = fx + int(0.15 * fw), fx + int(0.4 * fw)
            rc_x1, rc_x2 = fx + int(0.6 * fw), fx + int(0.85 * fw)
            mask[max(0, ck_y1):min(h, ck_y2), max(0, lc_x1):min(w, lc_x2)] = 1
            mask[max(0, ck_y1):min(h, ck_y2), max(0, rc_x1):min(w, rc_x2)] = 1
            
            mask_bool = mask.astype(bool)
            if not np.any(mask_bool):
                return np.mean(rgb_frame, axis=(0, 1))
            return np.mean(rgb_frame[mask_bool], axis=0)

        # MediaPipe Processing
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        all_roi_indices = FOREHEAD_LANDMARKS + LEFT_CHEEK_LANDMARKS + RIGHT_CHEEK_LANDMARKS
        pixel_coords = np.int32(pts[all_roi_indices, :2] * [w, h])

        # Convex hull mask over facial ROIs
        hull = cv2.convexHull(pixel_coords)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)

        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return np.mean(rgb_frame, axis=(0, 1))

        mean_rgb = np.mean(rgb_frame[mask_bool], axis=0)
        return mean_rgb

    def extract_video_rgb(self, video_path: str, save_path: str = None):
        """
        Processes an entire video file and returns extracted mean RGB signals over time and FPS.
        Optionally saves extracted signals to disk (.csv or .npy).
        
        Returns:
            rgb_signals: np.ndarray of shape (N, 3) where N is number of valid face frames
            fps: float sampling rate of the video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        rgb_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            mean_rgb = self.process_frame(frame)
            if mean_rgb is not None:
                rgb_list.append(mean_rgb)

        cap.release()
        rgb_signals = np.array(rgb_list)

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            if save_path.endswith(".csv"):
                np.savetxt(save_path, rgb_signals, delimiter=",", header="R,G,B", comments="")
                print(f"Saved RGB signals to CSV: {save_path}")
            elif save_path.endswith(".npy"):
                np.save(save_path, rgb_signals)
                print(f"Saved RGB signals to NPY: {save_path}")

        return rgb_signals, fps

    def close(self):
        if hasattr(self, "landmarker"):
            self.landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mean RGB time-series from facial video using MediaPipe ROI landmarks.")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file.")
    parser.add_argument("--output", type=str, default="results/subject49_rgb.csv", help="Path to save output CSV file.")
    args = parser.parse_args()

    print("Initializing FaceLandmarkExtractor...")
    try:
        extractor = FaceLandmarkExtractor()
        print("FaceLandmarkExtractor initialized successfully!")
        
        video_path = args.video
        if video_path is None:
            # Fallback to default locations if available
            sample_video = os.path.join("data", "UBFC-rPPG", "DATASET1", "subject1", "vid-001.avi")
            if os.path.exists(sample_video):
                video_path = sample_video
            else:
                print("\nError: No video path provided. Use --video 'path/to/video.avi'")
                sys.exit(1)

        print(f"\nProcessing video: {video_path}...")
        rgb_signals, fps = extractor.extract_video_rgb(video_path, save_path=args.output)
        print(f"Extracted {len(rgb_signals)} frames at {fps:.2f} FPS")
        print(f"RGB array shape: {rgb_signals.shape} (Frames, Channels: [R, G, B])")
        print("\nFirst 5 extracted RGB mean values:")
        for i, (r, g, b) in enumerate(rgb_signals[:5]):
            print(f"  Frame {i:02d}: R={r:.2f}, G={g:.2f}, B={b:.2f}")

        extractor.close()
    except Exception as e:
        import traceback
        print(f"Error during extraction: {e}")
        traceback.print_exc()


