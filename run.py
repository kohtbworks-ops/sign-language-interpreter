#!/usr/bin/env python3
"""Main entry point for the Sign Language Interpreter"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import SignLanguageApp

if __name__ == "__main__":
    try:
        app = SignLanguageApp()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()