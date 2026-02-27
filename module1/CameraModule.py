import cv2
import mediapipe as mp
import numpy as np
import time
import os

class CameraModule:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Thresholds
        self.YAW_THRESHOLD = 5.0
        self.ROLL_THRESHOLD = 5.0
        self.CENTER_THRESHOLD = 0.10
        self.AREA_THRESHOLD = 0.25
        self.STABLE_DURATION = 1.0
        
        # State tracking
        self.aligned_start_time = None
        self.captured = False
        self.captured_frame = None
        self.captured_bbox = None
        
        # Output setup
        self.output_path = "module1/output"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    
    def calculate_head_pose(self, landmarks, frame_width, frame_height):
        """Calculate yaw and roll angles from facial landmarks"""
        # Key landmarks: left eye, right eye, nose tip
        left_eye = landmarks[33]  # Left eye outer corner
        right_eye = landmarks[263]  # Right eye outer corner
        nose_tip = landmarks[1]  # Nose tip
        
        # Convert normalized coordinates to pixel coordinates
        left_eye_x = left_eye.x * frame_width
        left_eye_y = left_eye.y * frame_height
        right_eye_x = right_eye.x * frame_width
        right_eye_y = right_eye.y * frame_height
        nose_x = nose_tip.x * frame_width
        nose_y = nose_tip.y * frame_height
        
        # Calculate roll (head tilt)
        eye_delta_x = right_eye_x - left_eye_x
        eye_delta_y = right_eye_y - left_eye_y
        roll = np.degrees(np.arctan2(eye_delta_y, eye_delta_x))
        
        # Calculate yaw (head rotation left/right)
        eye_center_x = (left_eye_x + right_eye_x) / 2
        yaw = (nose_x - eye_center_x) / (frame_width / 2) * 30  # Approximate yaw
        
        return yaw, roll
    
    def get_face_bounding_box(self, landmarks, frame_width, frame_height):
        """Get bounding box coordinates from landmarks"""
        x_coords = [lm.x * frame_width for lm in landmarks]
        y_coords = [lm.y * frame_height for lm in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        return x_min, y_min, x_max, y_max
    
    def is_face_centered(self, bbox, frame_width, frame_height):
        """Check if face is centered in frame"""
        x_min, y_min, x_max, y_max = bbox
        face_center_x = (x_min + x_max) / 2
        face_center_y = (y_min + y_max) / 2
        
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        x_offset = abs(face_center_x - frame_center_x) / frame_width
        y_offset = abs(face_center_y - frame_center_y) / frame_height
        
        return x_offset < self.CENTER_THRESHOLD and y_offset < self.CENTER_THRESHOLD
    
    def is_face_large_enough(self, bbox, frame_width, frame_height):
        """Check if face occupies at least 25% of frame"""
        x_min, y_min, x_max, y_max = bbox
        face_area = (x_max - x_min) * (y_max - y_min)
        frame_area = frame_width * frame_height
        
        return (face_area / frame_area) >= self.AREA_THRESHOLD
    
    def run(self):
        """Main capture loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Camera started. Align your face to capture...")
        print("Press 'q' to quit")
        
        while cap.isOpened() and not self.captured:
            success, frame = cap.read()
            if not success:
                break
            
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            is_aligned = False
            status_text = "NO FACE DETECTED"
            box_color = (0, 0, 255)  # Red
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                
                # Calculate head pose
                yaw, roll = self.calculate_head_pose(landmarks, frame_width, frame_height)
                
                # Get bounding box
                bbox = self.get_face_bounding_box(landmarks, frame_width, frame_height)
                x_min, y_min, x_max, y_max = bbox
                
                # Check all conditions
                pose_aligned = abs(yaw) < self.YAW_THRESHOLD and abs(roll) < self.ROLL_THRESHOLD
                centered = self.is_face_centered(bbox, frame_width, frame_height)
                large_enough = self.is_face_large_enough(bbox, frame_width, frame_height)
                
                is_aligned = pose_aligned and centered and large_enough
                
                # Draw bounding box
                if is_aligned:
                    box_color = (0, 255, 0)  # Green
                    status_text = "ALIGNED - HOLD STILL"
                else:
                    status_messages = []
                    if not pose_aligned:
                        status_messages.append(f"Pose: Y={yaw:.1f} R={roll:.1f}")
                    if not centered:
                        status_messages.append("Not centered")
                    if not large_enough:
                        status_messages.append("Move closer")
                    status_text = " | ".join(status_messages)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            
            # Handle alignment timing
            current_time = time.time()
            if is_aligned:
                if self.aligned_start_time is None:
                    self.aligned_start_time = current_time
                elif current_time - self.aligned_start_time >= self.STABLE_DURATION:
                    # Capture image
                    timestamp = int(time.time())
                    filename = f"{self.output_path}/capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Captured: {filename}")
                    self.captured = True
                    self.captured_frame = frame.copy()
                    self.captured_bbox = bbox
                    break
                else:
                    # Show countdown
                    elapsed = current_time - self.aligned_start_time
                    remaining = self.STABLE_DURATION - elapsed
                    status_text = f"CAPTURING IN {remaining:.1f}s"
            else:
                self.aligned_start_time = None
            
            # Display status
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
            cv2.imshow('Camera Module', frame)
            
            # Exit on 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("Exiting...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Return captured data
        if self.captured:
            return {
                "image": self.captured_frame,
                "bbox": self.captured_bbox,
                "success": True
            }
        else:
            return {
                "image": None,
                "bbox": None,
                "success": False
            }

if __name__ == "__main__":
    camera = CameraModule()
    camera.run()
