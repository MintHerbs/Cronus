import cv2
import mediapipe as mp
import numpy as np
import time
import os

class CameraModule:
    def __init__(self):
        # ── New MediaPipe 0.10+ API ──────────────────────────────────────────
        BaseOptions   = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        # Download model if not present
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "face_landmarker.task"
        )
        if not os.path.exists(self.model_path):
            self._download_model()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Thresholds
        self.YAW_THRESHOLD    = 5.0
        self.ROLL_THRESHOLD   = 5.0
        self.CENTER_THRESHOLD = 0.10
        self.AREA_THRESHOLD   = 0.25
        self.STABLE_DURATION  = 1.0

        # State
        self.aligned_start_time = None
        self.captured           = False
        self.captured_frame     = None
        self.captured_bbox      = None

        # Output folder
        self.output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output"
        )
        os.makedirs(self.output_path, exist_ok=True)

    # ── Model download ────────────────────────────────────────────────────────

    def _download_model(self):
        import urllib.request
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        print(f"Downloading MediaPipe face landmarker model…")
        urllib.request.urlretrieve(url, self.model_path)
        print(f"Saved → {self.model_path}")

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def calculate_head_pose(self, landmarks, frame_width, frame_height):
        left_eye  = landmarks[33]
        right_eye = landmarks[263]
        nose_tip  = landmarks[1]

        lx = left_eye.x  * frame_width
        ly = left_eye.y  * frame_height
        rx = right_eye.x * frame_width
        ry = right_eye.y * frame_height
        nx = nose_tip.x  * frame_width

        roll         = np.degrees(np.arctan2(ry - ly, rx - lx))
        eye_center_x = (lx + rx) / 2
        yaw          = (nx - eye_center_x) / (frame_width / 2) * 30
        return yaw, roll

    def get_face_bounding_box(self, landmarks, frame_width, frame_height):
        xs = [lm.x * frame_width  for lm in landmarks]
        ys = [lm.y * frame_height for lm in landmarks]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def is_face_centered(self, bbox, frame_width, frame_height):
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        return (
            abs(cx - frame_width  / 2) / frame_width  < self.CENTER_THRESHOLD and
            abs(cy - frame_height / 2) / frame_height < self.CENTER_THRESHOLD
        )

    def is_face_large_enough(self, bbox, frame_width, frame_height):
        x_min, y_min, x_max, y_max = bbox
        face_area  = (x_max - x_min) * (y_max - y_min)
        frame_area = frame_width * frame_height
        return (face_area / frame_area) >= self.AREA_THRESHOLD

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return {"image": None, "bbox": None, "success": False}

        print("Camera started. Align your face to capture…")
        print("Press 'q' to quit")

        while cap.isOpened() and not self.captured:
            success, frame = cap.read()
            if not success:
                break

            # Save a clean copy BEFORE any drawing happens
            clean_frame = frame.copy()

            frame_h, frame_w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            detection_result = self.landmarker.detect(mp_image)

            is_aligned  = False
            status_text = "NO FACE DETECTED"
            box_color   = (0, 0, 255)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]

                yaw, roll = self.calculate_head_pose(landmarks, frame_w, frame_h)
                bbox      = self.get_face_bounding_box(landmarks, frame_w, frame_h)
                x_min, y_min, x_max, y_max = bbox

                pose_aligned  = abs(yaw) < self.YAW_THRESHOLD and abs(roll) < self.ROLL_THRESHOLD
                centered      = self.is_face_centered(bbox, frame_w, frame_h)
                large_enough  = self.is_face_large_enough(bbox, frame_w, frame_h)
                is_aligned    = pose_aligned and centered and large_enough

                if is_aligned:
                    box_color   = (0, 255, 0)
                    status_text = "ALIGNED - HOLD STILL"
                else:
                    msgs = []
                    if not pose_aligned:  msgs.append(f"Pose: Y={yaw:.1f} R={roll:.1f}")
                    if not centered:      msgs.append("Not centered")
                    if not large_enough:  msgs.append("Move closer")
                    status_text = " | ".join(msgs)

                # Draw on `frame` only — for display purposes
                # clean_frame stays untouched
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

            current_time = time.time()
            if is_aligned:
                if self.aligned_start_time is None:
                    self.aligned_start_time = current_time
                elif current_time - self.aligned_start_time >= self.STABLE_DURATION:
                    timestamp = int(time.time())
                    filename  = f"{self.output_path}/capture_{timestamp}.jpg"
                    # Use clean_frame — no rectangle or text drawn on it
                    cv2.imwrite(filename, clean_frame)
                    print(f"Captured: {filename}")
                    self.captured       = True
                    self.captured_frame = clean_frame
                    self.captured_bbox  = bbox
                    break
                else:
                    remaining   = self.STABLE_DURATION - (current_time - self.aligned_start_time)
                    status_text = f"CAPTURING IN {remaining:.1f}s"
            else:
                self.aligned_start_time = None

            # Draw status text on `frame` only — for display purposes
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            cv2.imshow("Camera Module", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                print("Exiting…")
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.captured:
            return {
                "image":   self.captured_frame,
                "bbox":    self.captured_bbox,
                "success": True,
            }
        return {"image": None, "bbox": None, "success": False}


if __name__ == "__main__":
    camera = CameraModule()
    camera.run()