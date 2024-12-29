from src.capture import capture_frames
from src.model import load_yolo_model

if __name__ == "__main__":
    model = load_yolo_model()
    capture_frames(model)
