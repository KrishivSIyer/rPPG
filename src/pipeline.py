import os
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
    """MediaPipe Face Landmarker wrapper for rPPG ROI extraction."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default location relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model file not found at: {model_path}\n"
                "Please run script to download 'face_landmarker.task'."
            )

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

    def extract_landmarks(self, frame_bgr: np.ndarray):
        """
        Extract normalized facial landmarks (x, y, z) from a BGR image frame.
        Returns array of shape (478, 3) or None if no face detected.
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        return pts

    def extract_roi_mean_rgb(self, frame_bgr: np.ndarray, landmarks: np.ndarray):
        """
        Compute mean RGB pixel values across forehead and cheek ROIs.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        all_roi_indices = FOREHEAD_LANDMARKS + LEFT_CHEEK_LANDMARKS + RIGHT_CHEEK_LANDMARKS
        pixel_coords = np.int32(landmarks[all_roi_indices, :2] * [w, h])

        # Convex hull mask over facial ROIs
        hull = cv2.convexHull(pixel_coords)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)

        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return np.mean(rgb_frame, axis=(0, 1))

        mean_rgb = np.mean(rgb_frame[mask_bool], axis=0)
        return mean_rgb

    def close(self):
        self.landmarker.close()



