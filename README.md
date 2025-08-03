# Hailo AI BlueIris Object Detection

A Python application that integrates BlueIris camera feeds with Hailo AI accelerator for real-time object detection on Raspberry Pi.

## Features

- **Real-time Object Detection**: Uses Hailo AI accelerator for fast inference
- **BlueIris Integration**: Connects to BlueIris server to retrieve camera images
- **Configurable Detection**: Adjustable confidence thresholds and NMS parameters
- **Result Processing**: Saves images with bounding boxes and detection data
- **Webhook Support**: Optional webhook notifications for detected objects
- **Async Processing**: Efficient concurrent operations using asyncio

## Requirements

- Raspberry Pi with Hailo AI Hat
- Python 3.8+
- BlueIris server
- Hailo SDK installed
- Install dependencies: `pip install -r requirements.txt`

## How to run the project
source venv/bin/activate && python main.py --config config.yaml
python main.py --config config.yaml