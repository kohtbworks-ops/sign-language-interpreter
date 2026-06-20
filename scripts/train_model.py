"""Standalone script for training the model"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.interpreter import SignLanguageInterpreter
from src.utils import load_training_data

def main():
    print("🤖 Training Model")
    print("="*40)
    
    interpreter = SignLanguageInterpreter()
    
    # Load training data
    try:
        X, y = load_training_data("models/training_data.pkl")
        print(f"Loaded {len(X)} training samples")
        print(f"Labels: {np.unique(y)}")
        
        # Train model
        success = interpreter.train_model(X, y, epochs=50)
        
        if success:
            print("✅ Model trained and saved successfully!")
        else:
            print("❌ Training failed!")
            
    except FileNotFoundError:
        print("❌ No training data found!")
        print("Please record some gestures first using the main application.")
    
if __name__ == "__main__":
    main()