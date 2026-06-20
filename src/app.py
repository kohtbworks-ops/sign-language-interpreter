"""Main application with GUI"""

import cv2
import numpy as np
import time
from collections import Counter
from .interpreter import SignLanguageInterpreter
from .config import Config

class SignLanguageApp:
    def __init__(self):
        self.interpreter = SignLanguageInterpreter()
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        # Training data storage
        self.training_data = []
        self.training_labels = []
        
        # UI state
        self.current_mode = "predict"
        self.current_gesture_label = ""
        self.show_help = Config.SHOW_HELP
        self.last_predict_time = 0
        self.fps = 0
        
        # Colors
        self.colors = {
            'bg': (255, 255, 255),
            'text': (0, 0, 0),
            'prediction': (0, 255, 0),
            'recording': (0, 0, 255),
            'ui': (200, 200, 200),
            'warning': (0, 0, 255)
        }
    
    def draw_ui(self, frame, prediction=None, confidence=None):
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for UI
        overlay = frame.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 70), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Title
        cv2.putText(frame, "SIGN LANGUAGE INTERPRETER", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # MODE INDICATOR - REMOVED (comment these lines)
        # mode_color = (0, 255, 0) if self.current_mode == 'predict' else (0, 0, 255)
        # mode_text = f"Mode: {'PREDICT' if self.current_mode == 'predict' else 'RECORD'}"
        # cv2.putText(frame, mode_text, 
        #            (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # FPS display
        if Config.FPS_DISPLAY:
            cv2.putText(frame, f"FPS: {self.fps}", 
                       (w - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Prediction display (bottom area)
        if prediction and self.current_mode == 'predict':
            # Background box
            cv2.rectangle(frame, (10, h-90), (w-10, h-10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, h-90), (w-10, h-10), (0, 255, 0), 2)
            
            # Prediction text
            text = f"Gesture: {prediction}"
            cv2.putText(frame, text, (30, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            if confidence:
                conf_text = f"Confidence: {confidence:.2%}"
                cv2.putText(frame, conf_text, (w-280, h-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Recording status
        if self.interpreter.recording:
            # Blinking recording indicator
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (w-40, 40), 20, (0, 0, 255), -1)
            cv2.putText(frame, f"RECORDING: {self.current_gesture_label}", 
                       (w-300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Help text (bottom-left)
        if self.show_help:
            help_texts = [
                "Controls:",
                "p - Predict  |  r - Record  |  1-9 - Set label",
                "t - Train  |  s - Save  |  l - Load",
                "h - Toggle help  |  q - Quit"
            ]
            y_pos = h - 10
            for text in help_texts[::-1]:
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos -= 22
        
        # Hand detection indicator
        cv2.putText(frame, "Show your hand to the camera", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*50)
        print("SIGN LANGUAGE INTERPRETER")
        print("="*50)
        print("Controls:")
        print("1-9: Set gesture label")
        print("r: Record gesture")
        print("t: Train model")
        print("p: Predict mode")
        print("s: Save model")
        print("l: Load model")
        print("q: Quit")
        print("="*50 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, landmarks, has_hand = self.interpreter.preprocess_frame(frame)
            
            prediction = None
            confidence = None
            
            if self.current_mode == "predict" and has_hand:
                # Predict gesture
                prediction, confidence = self.interpreter.predict_gesture(landmarks)
                
                # Smooth predictions
                if prediction:
                    self.interpreter.prediction_history.append(prediction)
                    
                    # Get most common prediction in history
                    if len(self.interpreter.prediction_history) >= Config.SMOOTHING_WINDOW:
                        most_common = Counter(self.interpreter.prediction_history).most_common(1)[0]
                        if most_common[1] >= Config.SMOOTHING_WINDOW // 2:
                            prediction = most_common[0]
            
            elif self.current_mode == "record" and has_hand:
                # Record gesture for training
                if self.interpreter.record_frame(landmarks):
                    data, label = self.interpreter.get_recorded_data()
                    if data is not None:
                        self.training_data.extend(data)
                        self.training_labels.extend([label] * len(data))
                        print(f"Added {len(data)} frames for gesture '{label}'")
                        self.interpreter.recorded_landmarks = []
            
            # Draw UI
            frame = self.draw_ui(frame, prediction, confidence)
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_predict_time > 0:
                self.fps = int(1 / (current_time - self.last_predict_time + 0.001))
            self.last_predict_time = current_time
            
            # Display frame
            cv2.imshow("Sign Language Interpreter", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                self.current_mode = "predict"
                print("Switched to PREDICT mode")
            elif key == ord('r'):
                self.current_mode = "record"
                print("Switched to RECORD mode")
                if self.current_gesture_label:
                    self.interpreter.start_recording(self.current_gesture_label)
                else:
                    print("Please set a gesture label first (press 1-9)")
            elif key in [ord(str(i)) for i in range(1, 10)]:
                label = chr(key).upper()
                self.current_gesture_label = f"Gesture_{label}"
                print(f"Gesture label set to: {self.current_gesture_label}")
                if self.current_mode == "record":
                    self.interpreter.start_recording(self.current_gesture_label)
            elif key == ord('t'):
                if len(self.training_data) > 0:
                    print("Training model...")
                    success = self.interpreter.train_model(
                        self.training_data, 
                        self.training_labels
                    )
                    if success:
                        self.training_data = []
                        self.training_labels = []
                        print("Model trained successfully!")
                    else:
                        print("Training failed!")
                else:
                    print("No training data available. Record some gestures first!")
            elif key == ord('s'):
                if self.interpreter.model_loaded:
                    self.interpreter.save_model()
                else:
                    print("No model to save!")
            elif key == ord('l'):
                if self.interpreter.load_model():
                    print("Model loaded!")
                else:
                    print("No saved model found!")
            elif key == ord('h'):
                self.show_help = not self.show_help
                print(f"Help {'shown' if self.show_help else 'hidden'}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.interpreter.cleanup()
        print("Application closed")