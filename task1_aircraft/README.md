# Aircraft Detection

This task uses a pre-trained Faster R-CNN model to detect aircraft in images. The model is built using PyTorch and torchvision, and detects airplanes using bounding boxes.

# Task Description

- Load a static aircraft image
- Run object detection to identify aircraft
- Draw bounding boxes on detected aircraft
- Display the result and optionally save it

# Methodology

- Used 'fasterrcnn_resnet50_fpn' from torchvision models
- Loaded an image using OpenCV ('cv2')
- Converted to a PyTorch tensor
- Ran inference with the model
- Filtered predictions for "airplane" (COCO class 5)
- Drew bounding boxes and confidence score on the image
- The output will be displayed in a window, and saved to output_detected.jpg

# Files Included

- 'aircraft.py': Python script that loads an image and detects aircraft
- 'sample_images': Folder containing test images
- 'requirements.txt': Required packages
- 'README.md': Project documentation
- 'output_detected.jpg': Output
