# StreamClean: Real-time Background Removal for Linux Streaming

**StreamClean** is a real-time background removal tool for streaming applications like OBS. Inspired by Nvidia Broadcast, this project provides a quick solution for Linux users to remove their background during live streaming.

### Motivation

Nvidia Broadcast offers a fantastic background removal feature, but it currently doesn't support Linux. To fill this gap, I tried using the OBS background removal plugin, but I faced compatibility issues, especially with my RTX 3080 and TensorRT, possibly due to the system architecture and environment the app was built for. This led to the creation of StreamClean as a Linux-compatible alternative.

Special thanks to [@lef-fan Eleftherios Fanioudakis](https://github.com/lef-fan) for the inspiration behind this project. Eleftherios is working on a similar project in C++, which is expected to deliver much better performance.

### Features

- Real-time background removal using YOLOv8 segmentation.
- Works seamlessly with OBS via a virtual webcam output (using `pyvirtualcam`).
- Blue background for easy chroma keying in OBS.
- Optimized for Linux systems using Nvidia GPUs.

### Prerequisites

Before you can use StreamClean, make sure you have the following installed:

- **Linux** (Ubuntu/ParrotOS/Arch-based distros recommended).
- **Python 3.7+**.
- **Nvidia GPU** with support for CUDA.
- **v4l2loopback** kernel module (to create a virtual webcam).

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/StreamClean.git
   cd StreamClean
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt

   ```

3. **Install v4l2loopback (for virtual webcam)**:
   For Debian-based distros (e.g., Ubuntu, ParrotOS):

   ```bash
   sudo apt-get install v4l2loopback-dkms
   ```

   For Arch Linux:

   ```bash
   sudo pacman -S v4l2loopback-dkms
   ```

4. **Load the v4l2loopback module**:

   ```bash
   sudo modprobe v4l2loopback
   ```

5. **Download the yolov9e-seg.pt**:
   i used yolov9 form ultralitics!
   ```bash
      https://huggingface.co/merve/yolov9
   ```

5.1. **Run the main.py file**:

```bash
    python3 main.py
```

### Libraries Used

    - **YOLOv8** (via the ultralytics library) for real-time object detection and segmentation.
    - **OpenCV** for image processing (including background removal).
    - **NumPy** for efficient array manipulation.
    - **PyVirtualCam** for creating a virtual webcam.
    - **v4l2loopback** kernel module for virtual camera support on Linux.
