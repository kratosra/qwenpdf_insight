# scripts/launch_unified.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.qwen_interface_unified import demo

if __name__ == "__main__":
    demo.launch()
