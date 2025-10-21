import cv2
import mediapipe as mp
import numpy as np

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def recognize_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        if (thumb_tip.y < thumb_mcp.y and
            index_tip.y > index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y):
            return "Thumbs Up"
        
        if (index_tip.y < index_mcp.y and
            middle_tip.y < middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y and
            thumb_tip.x > thumb_mcp.x):
            return "Peace Sign"
        
        fingers_extended = 0
        if index_tip.y < index_mcp.y: fingers_extended += 1
        if middle_tip.y < middle_mcp.y: fingers_extended += 1
        if ring_tip.y < ring_mcp.y: fingers_extended += 1
        if pinky_tip.y < pinky_mcp.y: fingers_extended += 1
        
        if fingers_extended >= 3:
            return "Hello Wave"
        
        return "No Gesture"
    
    def draw_landmarks_with_instructions(self, frame, gesture_text):
        cv2.putText(frame, f"Gesture: {gesture_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, "Make sure your entire hand is visible", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        instructions = [
            "Show one of these gestures:",
            "- Open hand (Wave)",
            "- Peace sign", 
            "- Thumbs up"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, 
                       (10, 90 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press 'Q' to quit", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting Gesture Recognition...")
        print("Show these gestures to the camera:")
        print("- Open hand/Wave")
        print("- Peace sign") 
        print("- Thumbs up")
        print("Press 'Q' to quit the application")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            gesture_text = "No Gesture Detected"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append(landmark)
                    
                    gesture_text = self.recognize_gesture(landmarks)
                    
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in landmarks]
                    y_coords = [lm.y * h for lm in landmarks]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            self.draw_landmarks_with_instructions(frame, gesture_text)
            
            cv2.imshow('Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting application...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Gesture recognition stopped.")

def main():
    try:
        recognizer = GestureRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()