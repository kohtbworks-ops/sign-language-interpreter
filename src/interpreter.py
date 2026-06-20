"""Core sign language interpreter class"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import json
import os
from collections import deque
from .config import Config
from .utils import normalize_landmarks

class SignLanguageInterpreter:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=Config.MAX_NUM_HANDS,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # For smoothing predictions
        self.prediction_history = deque(maxlen=Config.PREDICTION_HISTORY_SIZE)
        self.last_prediction = ""
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        
        # Initialize model
        self.model = None
        self.label_encoder = LabelEncoder()
        self.labels = []
        self.model_loaded = False
        self.feature_extractor = None
        
        # Recording state
        self.recording = False
        self.recorded_landmarks = []
        self.recorded_label = ""
        self.recording_frames = 0
        self.max_recording_frames = Config.MAX_RECORDING_FRAMES
        
        # Create directories
        Config.create_directories()
        
        # Try to load existing model
        self.load_model()
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks as feature vector"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def preprocess_frame(self, frame):
        """Process frame and extract hand landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Draw landmarks on frame
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Extract landmarks from first hand
            landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
            
            # Normalize landmarks
            normalized_landmarks = normalize_landmarks(landmarks)
            
            return frame, normalized_landmarks, True
        else:
            return frame, None, False
    
    def create_model(self, num_classes):
        """Create a neural network for gesture classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def save_model(self, model_path=None, labels_path=None):
        """Save trained model and labels"""
        if not self.model:
            print("No model to save!")
            return False
        
        if model_path is None:
            model_path = Config.MODEL_PATH
        if labels_path is None:
            labels_path = Config.LABELS_PATH
        
        try:
            self.model.save(model_path)
            with open(labels_path, 'w') as f:
                json.dump({'labels': self.labels.tolist()}, f)
            print(f"✅ Model saved to {model_path}")
            print(f"✅ Labels saved to {labels_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_path=None, labels_path=None):
        """Load trained model and labels"""
        if model_path is None:
            model_path = Config.MODEL_PATH
        if labels_path is None:
            labels_path = Config.LABELS_PATH
        
        try:
            if os.path.exists(model_path) and os.path.exists(labels_path):
                self.model = tf.keras.models.load_model(model_path)
                with open(labels_path, 'r') as f:
                    data = json.load(f)
                    self.labels = np.array(data['labels'])
                    self.label_encoder.fit(self.labels)
                self.model_loaded = True
                print(f"✅ Model loaded successfully from {model_path}")
                print(f"📝 Recognized gestures: {list(self.labels)}")
                return True
            else:
                print("ℹ️ No saved model found. Please train a model first.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def train_model(self, X, y, epochs=None, batch_size=None):
        """Train the model on recorded gestures"""
        if epochs is None:
            epochs = Config.EPOCHS
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
        
        if len(X) < 10:
            print("❌ Not enough training data! Need at least 10 samples.")
            return False
        
        # Prepare data
        X = np.array(X)
        self.labels = np.unique(y)
        self.label_encoder.fit(self.labels)
        y_encoded = self.label_encoder.transform(y)
        y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=len(self.labels))
        
        print(f"📊 Training with {len(X)} samples, {len(self.labels)} classes")
        
        # Create and train model
        self.model = self.create_model(len(self.labels))
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X, y_one_hot,
            epochs=epochs,
            batch_size=min(batch_size, len(X)),
            validation_split=Config.VALIDATION_SPLIT,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.model_loaded = True
        self.save_model()
        
        print(f"✅ Model trained on {len(self.labels)} gestures!")
        print(f"📈 Final training accuracy: {history.history['accuracy'][-1]:.2f}")
        if history.history.get('val_accuracy'):
            print(f"📈 Final validation accuracy: {history.history['val_accuracy'][-1]:.2f}")
        
        return True
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks"""
        if not self.model_loaded or landmarks is None:
            return None, 0
        
        try:
            # Reshape for prediction
            landmarks_array = np.array(landmarks).reshape(1, -1)
            
            # Predict
            predictions = self.model.predict(landmarks_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            if confidence > self.confidence_threshold:
                gesture = self.label_encoder.inverse_transform([class_idx])[0]
                return gesture, confidence
            return None, confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0
    
    def start_recording(self, label):
        """Start recording a new gesture"""
        self.recording = True
        self.recorded_landmarks = []
        self.recorded_label = label
        self.recording_frames = 0
        print(f"🎥 Recording gesture: {label} (hold gesture for {self.max_recording_frames} frames)")
    
    def record_frame(self, landmarks):
        """Record a frame for training"""
        if self.recording and landmarks is not None:
            self.recorded_landmarks.append(landmarks)
            self.recording_frames += 1
            
            if self.recording_frames >= self.max_recording_frames:
                self.recording = False
                print(f"✅ Finished recording {self.recorded_label}")
                return True
        return False
    
    def get_recorded_data(self):
        """Get recorded gesture data for training"""
        if self.recorded_landmarks:
            return np.array(self.recorded_landmarks), self.recorded_label
        return None, None
    
    def cleanup(self):
        """Clean up resources"""
        self.hands.close()