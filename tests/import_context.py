"""Script to import from parent directory"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
print(sys.path)