# Intrusion Detection System

## Overview

The Intrusion Detection System is designed to monitor  environments using computer vision techniques. It can process real-time video feeds or uploaded images to detect potential intrusions and other relevant activities. The system utilizes YOLOv8, a state-of-the-art object detection model, to analyze video streams and images for detection purposes.

## Features

- **Real-Time Video Feed**: Monitors live video from a webcam or camera for immediate detection.
- **Image Upload**: Allows users to upload images for detection.
- **Intrusion Detection**: Utilizes YOLOv8 for accurate detection of intruders and relevant objects.
- **User-Friendly Interface**: Simple and intuitive interface for selecting video or image upload options.

## Technologies Used

- **Flask**: Web framework for building the application.
- **OpenCV**: Library for computer vision tasks.
- **YOLOv8**: Object detection model used for analyzing video and images.
- **HTML/CSS/JavaScript**: Frontend technologies for building the user interface.

## Installation

### Prerequisites

- Python 3.9

### Clone the Repository

```bash
git clone https://github.com/yourusername/intrusion-detection.git
```


### Install Dependencies

```bash
pip install -r requirements.txt
```

### Model File

Make sure to download the YOLOv8 model file (`yolov8n.pt`) and place it in the project directory.

## Running the Application

1. Start the Flask server:

```bash
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Choose between real-time video feed or image upload to detect intrusions.

## Usage

- **Real-Time Video Feed**: Click the "Real-Time Video Feed" button to start the video stream from your camera. Use the "Play" and "Pause" buttons to control the video feed.
- **Upload Image**: Click the "Upload Image" button to select an image file from your device and get detection results.
