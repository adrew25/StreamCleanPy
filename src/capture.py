import cv2
import threading
from src.postProcessing import MaskPostProcessor
from .model import process_frame
import pyvirtualcam

# Initialize the mask post-processor with desired parameters
post_processor = MaskPostProcessor(
    kernel_size_smooth=15, kernel_size_refine=5, alpha_blend=0.5
)

# Global variables for frame handling and thread control
current_frame = None
frame_lock = threading.Lock()
running = True


def capture_frames(model):
    """
    Capture frames from webcam, process them, and send them to OBS via virtual camera.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize pyvirtualcam
    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        print(f"Using virtual camera: {cam.device}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame (detect person, create mask, remove background)
            frame_rgba = process_frame(frame, model)

            # Convert RGBA to RGB for virtual camera (pyvirtualcam expects RGB format)
            frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2RGB)

            # Send the processed frame to the virtual camera
            cam.send(frame_rgb)

            # Optional: Show the processed frame in a window (can be removed if not needed)
            # cv2.imshow("Processed Frame", frame_rgba)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Capture thread stopped.")
