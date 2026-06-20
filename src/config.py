"""Configuration settings for the sign language interpreter"""

import os

class Config:
    # Model settings
    MODEL_PATH = "models/gesture_model.h5"
    LABELS_PATH = "models/labels.json"
    DATA_PATH = "models/training_data.pkl"
    
    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    MAX_NUM_HANDS = 2
    
    # Training settings
    EPOCHS = 30
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    MAX_RECORDING_FRAMES = 30
    
    # Prediction settings
    CONFIDENCE_THRESHOLD = 0.7
    PREDICTION_HISTORY_SIZE = 10
    SMOOTHING_WINDOW = 5
    
    # UI settings
    SHOW_HELP = True
    FPS_DISPLAY = True
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        dirs = ['models', 'data/training', 'data/test', 'output/logs', 'output/screenshots']
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)