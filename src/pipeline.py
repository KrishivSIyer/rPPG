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

# A1: Exclusion masks — eye and lip regions to subtract so the ROI mask
# covers forehead and cheeks ONLY (as documented).
LEFT_EYE_LANDMARKS  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                        173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                        466, 388, 387, 386, 385, 384, 398]
MOUTH_LANDMARKS     = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                        95, 185, 40, 39, 37, 0, 267, 269, 270, 409]


def _fill_hull(mask, pts, h, w, value=1):
    """Fill the convex hull of `pts` (pixel coords) into `mask`."""
    hull = cv2.convexHull(np.int32(pts))
    cv2.fillConvexPoly(mask, hull, value)


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

    def process_frame(self, frame_bgr: np.ndarray, save_mask_path: str = None):
        """
        Processes a single BGR frame.

        A1 fix: builds one convex hull per ROI region (forehead, left cheek,
        right cheek), takes their union, then subtracts eye and mouth pixels.
        This ensures the mask covers ONLY clean skin regions as documented.

        Returns:
            mean_rgb: np.ndarray of shape (3,) with average [R, G, B] values
                      over facial ROIs, or None if no face is detected.
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

            # Forehead ROI (top third of face bounding box)
            fh_y1, fh_y2 = fy + int(0.05 * fh), fy + int(0.28 * fh)
            fh_x1, fh_x2 = fx + int(0.25 * fw), fx + int(0.75 * fw)
            mask[max(0, fh_y1):min(h, fh_y2), max(0, fh_x1):min(w, fh_x2)] = 1

            # A1 fix: cheek boxes start BELOW the eye band (> 50% of face height)
            # to avoid including eye pixels in the OpenCV fallback path.
            ck_y1, ck_y2 = fy + int(0.52 * fh), fy + int(0.75 * fh)
            lc_x1, lc_x2 = fx + int(0.10 * fw), fx + int(0.38 * fw)
            rc_x1, rc_x2 = fx + int(0.62 * fw), fx + int(0.90 * fw)
            mask[max(0, ck_y1):min(h, ck_y2), max(0, lc_x1):min(w, lc_x2)] = 1
            mask[max(0, ck_y1):min(h, ck_y2), max(0, rc_x1):min(w, rc_x2)] = 1

            mask_bool = mask.astype(bool)
            if not np.any(mask_bool):
                return np.mean(rgb_frame, axis=(0, 1))
            return np.mean(rgb_frame[mask_bool], axis=0)

        # --- MediaPipe Processing ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

        # A1 fix: build one hull per skin region (forehead, left cheek, right
        # cheek) and fill each separately to avoid bridging the nose and eyes.
        mask = np.zeros((h, w), dtype=np.uint8)
        for indices in [FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS]:
            region_pts = pts[indices]
            _fill_hull(mask, region_pts, h, w, value=1)

        # A1 fix: subtract eye and mouth polygons from the combined skin mask.
        exclusion = np.zeros((h, w), dtype=np.uint8)
        for indices in [LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS, MOUTH_LANDMARKS]:
            region_pts = pts[indices]
            _fill_hull(exclusion, region_pts, h, w, value=1)
        mask = np.clip(mask - exclusion, 0, 1)

        # Optional debug: save mask overlay image for visual verification (A1)
        if save_mask_path is not None:
            overlay = rgb_frame.copy()
            overlay[mask.astype(bool)] = (overlay[mask.astype(bool)] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            cv2.imwrite(save_mask_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return None

        mean_rgb = np.mean(rgb_frame[mask_bool], axis=0)
        return mean_rgb

    def extract_video_rgb(self, video_path: str, save_path: str = None, save_mask_dir: str = None, mask_save_nth: int = 30):
        """
        Processes an entire video file and returns extracted mean RGB signals over time and FPS.
        Optionally saves extracted signals to disk (.csv or .npy).

        A1: Pass `save_mask_dir` to save overlay mask PNGs every `mask_save_nth` frames
            for visual verification that the mask covers forehead and cheeks only.

        A2 fix: One row is written per frame. Frames where no face is detected
        receive NaN for R, G, B. The CSV has columns frame_idx, time_s, R, G, B.
        Old R,G,B-only CSVs are still readable (time column treated as frame_idx/fps).

        Returns:
            rgb_signals: np.ndarray of shape (N, 5) [frame_idx, time_s, R, G, B]
            fps: float sampling rate of the video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        if save_mask_dir is not None:
            os.makedirs(save_mask_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        rows = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            time_s = frame_idx / fps

            # Save debug mask overlay every N frames if requested (A1 acceptance check)
            mask_path = None
            if save_mask_dir is not None and frame_idx % mask_save_nth == 0:
                mask_path = os.path.join(save_mask_dir, f"mask_{frame_idx:05d}.png")

            mean_rgb = self.process_frame(frame, save_mask_path=mask_path)

            if mean_rgb is not None:
                rows.append([frame_idx, time_s, mean_rgb[0], mean_rgb[1], mean_rgb[2]])
            else:
                # A2: keep one row per frame; write NaN for missing detections
                rows.append([frame_idx, time_s, float("nan"), float("nan"), float("nan")])

            frame_idx += 1

        cap.release()
        rgb_signals = np.array(rows, dtype=np.float64)  # shape (N, 5)

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            if save_path.endswith(".csv"):
                np.savetxt(
                    save_path, rgb_signals, delimiter=",",
                    header="frame_idx,time_s,R,G,B", comments=""
                )
                nan_count = int(np.isnan(rgb_signals[:, 2]).sum())
                print(f"Saved RGB signals to CSV: {save_path}  ({nan_count} NaN frames out of {len(rgb_signals)})")
            elif save_path.endswith(".npy"):
                np.save(save_path, rgb_signals)
                print(f"Saved RGB signals to NPY: {save_path}")

        return rgb_signals, fps

    def close(self):
        if hasattr(self, "landmarker"):
            self.landmarker.close()


def load_rgb_csv(csv_path: str, fps: float = 30.0):
    """
    A2: Reads a CSV that may be either the new format (frame_idx, time_s, R, G, B)
    or the old format (R, G, B only). Returns (rgb_array, time_array) where
    rgb_array has shape (N, 3) and NaN gaps shorter than 0.5 s are linearly
    interpolated. Longer NaN gaps remain NaN (callers should flag or drop them).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "R" not in df.columns:
        raise ValueError(f"CSV {csv_path} does not contain an 'R' column.")

    if "time_s" in df.columns:
        time_s = df["time_s"].values
    elif "frame_idx" in df.columns:
        time_s = df["frame_idx"].values / fps
    else:
        time_s = np.arange(len(df)) / fps

    rgb = df[["R", "G", "B"]].values.astype(np.float64)

    # A2: linearly interpolate NaN gaps shorter than 0.5 s
    gap_threshold = 0.5  # seconds
    for ch in range(3):
        col = rgb[:, ch]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        # Find contiguous NaN runs
        changes = np.diff(nan_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends   = np.where(changes == -1)[0] + 1
        if nan_mask[0]:
            starts = np.concatenate([[0], starts])
        if nan_mask[-1]:
            ends = np.concatenate([ends, [len(col)]])
        for s, e in zip(starts, ends):
            gap_dur = time_s[e - 1] - time_s[s] if e > s else 0
            if gap_dur <= gap_threshold:
                # Interpolate using surrounding valid samples
                left  = s - 1
                right = e
                if left >= 0 and right < len(col):
                    col[s:e] = np.interp(time_s[s:e], [time_s[left], time_s[right]], [col[left], col[right]])
        rgb[:, ch] = col

    return rgb.astype(np.float32), time_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mean RGB time-series from facial video using MediaPipe ROI landmarks.")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file.")
    parser.add_argument("--output", type=str, default="results/subject49_rgb.csv", help="Path to save output CSV file.")
    parser.add_argument("--save_mask_dir", type=str, default=None,
                        help="If set, save overlay mask PNGs to this directory for visual ROI verification (A1 debug).")
    parser.add_argument("--mask_save_nth", type=int, default=30,
                        help="Save mask overlay for every Nth frame (default 30).")
    args = parser.parse_args()

    print("Initializing FaceLandmarkExtractor...")
    try:
        extractor = FaceLandmarkExtractor()
        print("FaceLandmarkExtractor initialized successfully!")

        video_path = args.video
        if video_path is None:
            sample_video = os.path.join("data", "UBFC-rPPG", "DATASET1", "subject1", "vid-001.avi")
            if os.path.exists(sample_video):
                video_path = sample_video
            else:
                print("\nError: No video path provided. Use --video 'path/to/video.avi'")
                sys.exit(1)

        print(f"\nProcessing video: {video_path}...")
        rgb_signals, fps = extractor.extract_video_rgb(
            video_path,
            save_path=args.output,
            save_mask_dir=args.save_mask_dir,
            mask_save_nth=args.mask_save_nth,
        )
        valid = int((~np.isnan(rgb_signals[:, 2])).sum())
        print(f"Extracted {valid} valid frames out of {len(rgb_signals)} total at {fps:.2f} FPS")

        extractor.close()
    except Exception as e:
        import traceback
        print(f"Error during extraction: {e}")
        traceback.print_exc()
