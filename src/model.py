from ultralytics import YOLO
import cv2
import numpy as np
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)


def load_yolo_model():
    """
    Load the YOLO segmentation model.
    """
    return YOLO("models/yolov9e-seg.pt")


def detect_person(frame, model):
    """
    Detect persons in the frame using YOLO.
    """
    results = model(frame)
    detections = results[0]
    masks = detections.masks.data.cpu().numpy() if detections.masks is not None else []
    classes = detections.boxes.cls.cpu().numpy() if detections.boxes is not None else []
    confidences = (
        detections.boxes.conf.cpu().numpy() if detections.boxes is not None else []
    )

    # Create a list of person masks (class 0 is 'person' in COCO dataset)
    person_masks = [
        mask
        for mask, cls, conf in zip(masks, classes, confidences)
        if int(cls) == 0 and conf > 0.5
    ]
    return person_masks


def create_mask(frame, person_masks):
    """
    Create a binary mask for detected persons.
    """
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for person_mask in person_masks:
        person_mask_resized = cv2.resize(
            person_mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask[person_mask_resized > 0.5] = 255
    return mask


def remove_background_with_alpha(frame, mask, default_alpha=255):
    """
    Apply mask and create an alpha channel for the frame.
    """
    rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # Convert to RGBA
    # Use mask as the alpha channel
    rgba_frame[:, :, 3] = cv2.normalize(mask, None, 0, default_alpha, cv2.NORM_MINMAX)
    return rgba_frame


def process_frame(frame, model):
    """
    Process a single frame: detect person, create mask, and remove background.
    """
    # Detect person and create the segmentation mask
    person_masks = detect_person(frame, model)
    mask = create_mask(frame, person_masks)

    # Remove background and add alpha channel (use mask)
    frame_rgba = remove_background_with_alpha(frame, mask)

    # Replace the background (alpha = 0) with blue color (for chroma keying)
    frame_rgba[:, :, :3][frame_rgba[:, :, 3] == 0] = [255, 0, 0]  # Blue BG

    return frame_rgba
