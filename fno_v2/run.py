"""
Simple wrapper script to run main.py
"""
import os
import sys

# Add the necessary paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now import and run the main function from main.py
# First, modify sys.path to make the imports in main.py work
sys.path.insert(0, os.path.abspath('..'))

# Import the main function from main.py and run it
from fno_v2.main_direct import main

if __name__ == "__main__":
    main() 