"""Utility functions for the sign language interpreter"""

import cv2
import numpy as np
import json
import os
from datetime import datetime

def save_landmarks_to_file(landmarks, filename):
    """Save landmarks to a file"""
    np.save(filename, landmarks)
    print(f"Saved landmarks to {filename}")

def load_landmarks_from_file(filename):
    """Load landmarks from a file"""
    return np.load(filename)

def save_training_data(data, labels, filename):
    """Save training data and labels"""
    with open(filename, 'w') as f:
        json.dump({'data': data.tolist(), 'labels': labels.tolist()}, f)

def load_training_data(filename):
    """Load training data and labels"""
    with open(filename, 'r') as f:
        data = json.load(f)
        return np.array(data['data']), np.array(data['labels'])

def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def add_text_to_frame(frame, text, position, color=(255, 255, 255), size=0.7):
    """Add text with background for better visibility"""
    x, y = position
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2)
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                  (x-5, y-text_height-5), 
                  (x+text_width+5, y+5), 
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
    
    return frame

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist position"""
    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
    normalized = []
    
    for i in range(0, len(landmarks), 3):
        x = landmarks[i] - wrist_x
        y = landmarks[i+1] - wrist_y
        z = landmarks[i+2] - wrist_z
        normalized.extend([x, y, z])
    
    return normalized

def calculate_hand_features(landmarks):
    """Calculate additional hand features"""
    if landmarks is None or len(landmarks) < 63:
        return None
    
    features = []
    
    # Calculate distances between key points
    wrist = (landmarks[0], landmarks[1], landmarks[2])
    
    # Fingertip indices (approximate)
    finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
    finger_mcp = [5, 9, 13, 17]    # corresponding MCP joints
    
    # Calculate distances from wrist to fingertips
    for i, tip in enumerate(finger_tips):
        idx = tip * 3
        tip_point = (landmarks[idx], landmarks[idx+1], landmarks[idx+2])
        
        # Euclidean distance
        distance = np.sqrt(sum((tip_point[i] - wrist[i])**2 for i in range(3)))
        features.append(distance)
    
    # Calculate angles between fingers
    for i in range(4):
        for j in range(i+1, 4):
            idx1 = finger_tips[i] * 3
            idx2 = finger_tips[j] * 3
            
            point1 = np.array([landmarks[idx1], landmarks[idx1+1], landmarks[idx1+2]])
            point2 = np.array([landmarks[idx2], landmarks[idx2+1], landmarks[idx2+2]])
            
            angle = np.arccos(np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2) + 1e-6))
            features.append(angle)
    
    return np.array(features)