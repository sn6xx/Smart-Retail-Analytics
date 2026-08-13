# Smart Retail Analytics

**Trainee:** Saja Aljamal

## Capstone Project – Computer Vision for Developers with Ultralytics

A real-world computer vision application built with Ultralytics YOLO for smart retail analytics.

The project demonstrates an end-to-end computer vision pipeline for detecting people, applying video analytics, generating privacy-preserving outputs, and evaluating YOLO model performance.

---

## 1. Project Overview

### Problem

Retail environments generate large amounts of visual data that can be used to understand customer presence and movement.

Manual analysis of video footage can be time-consuming and difficult to scale. Computer vision can automate the detection and analysis of people in video streams.

### Solution

The Smart Retail Analytics system uses Ultralytics YOLO and OpenCV to process images and videos.

The implemented pipeline includes:

1. YOLO object detection.
2. Instance segmentation.
3. Real-world video analytics.
4. People counting.
5. Heatmap generation.
6. Privacy-preserving object blurring.
7. Video analytics.
8. Model validation and performance evaluation.

The project is designed as an end-to-end computer vision application rather than a standalone object detection demonstration.

---

## 2. Project Objectives

The main objectives of this capstone project are:

- Perform object detection using Ultralytics YOLO.
- Perform a computer vision task beyond plain detection using instance segmentation.
- Apply YOLO to a real-world video scenario.
- Analyze people movement in video.
- Count detected people.
- Generate movement heatmaps.
- Apply privacy-preserving object blurring.
- Perform additional video analytics.
- Evaluate the pretrained YOLO model using quantitative validation metrics.
- Document real execution results and generated outputs.
- Extend the project with custom model training and deployment/export.

---

## 3. Technologies

- Python
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- PyTorch
- Google Colab
- GitHub
- GitHub Desktop

---

## 4. Computer Vision Models

### 4.1 Object Detection Model

The project uses the pretrained YOLO model:

```text
yolo26n.pt
```

The model is used for object detection on images and video frames.

The detection model generates:

- Bounding boxes
- Class labels
- Confidence scores

### 4.2 Instance Segmentation Model

The project uses the task-specific segmentation model:

```text
yolo26n-seg.pt
```

Instance segmentation extends basic object detection by generating object masks in addition to bounding boxes and class predictions.

---

## 5. Core Vision Tasks & Inference

### 5.1 Load YOLO Model

A pretrained Ultralytics YOLO model is loaded using the Python API.

```python
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo26n.pt")

print("Model loaded successfully!")
```

The model was successfully loaded and prepared for inference.

### 5.2 YOLO Object Detection

A test frame is extracted from the input video using OpenCV.

```python
import cv2

video_path = "solutions_ci_demo.mp4"
image_path = "detection_test_frame.jpg"

cap = cv2.VideoCapture(video_path)

success, frame = cap.read()

if not success:
    raise RuntimeError("Could not read the input video.")

cv2.imwrite(image_path, frame)

cap.release()

print("Test frame saved as:", image_path)
```

The executed cell produced:

```text
Test frame saved as: detection_test_frame.jpg
```

### 5.3 Run YOLO Inference

The extracted frame is passed to the pretrained YOLO model.

```python
# Run YOLO inference
results = model(image_path)

# Generate annotated image
annotated_image = results[0].plot()

# Save the detection result
detection_output = "detection_result.jpg"

cv2.imwrite(detection_output, annotated_image)

print("Detection completed successfully!")
print("Output saved as:", detection_output)
```

Example execution output:

```text
1 image ... 19 persons ...
Detection completed successfully!
Output saved as: detection_result.jpg
```

The model successfully detected people in the test frame and generated an annotated image containing bounding boxes, class labels, and confidence information.

### 5.4 Instance Segmentation

Instance segmentation is included as the computer vision task beyond plain object detection.

The task-specific YOLO segmentation weights are:

```text
yolo26n-seg.pt
```

The segmentation model generates object masks in addition to detection information.

Example output:

```text
segmentation_result.jpg
```

The segmentation stage demonstrates the use of task-specific YOLO weights for a vision task beyond basic object detection.

---

## 6. Real-World Solution & Video Analytics

The project applies Ultralytics Solutions to a real video-processing pipeline.

The video-processing workflow is:

```text
Input Video
     |
     v
OpenCV Video Capture
     |
     v
YOLO / Ultralytics Solution
     |
     v
Frame Processing
     |
     v
Visualization / Analytics
     |
     v
OpenCV Video Writer
     |
     v
Output Video
```

This satisfies the real-world video analytics requirement by processing video frames through an OpenCV capture/process/write pipeline.

### 6.1 People Counting

Ultralytics video analytics functionality is used to detect and count people in the video stream.

The video is processed frame by frame using OpenCV, and the processed frames are written to an output video.

Example output:

```text
people_counting_result.mp4
```

### 6.2 Heatmap Generation

Ultralytics Heatmap functionality is used to visualize movement within the video.

The heatmap provides a visual representation of areas where detected people move through the scene.

Example output:

```text
heatmap_result.mp4
```

### 6.3 Object Blurring

The Ultralytics `ObjectBlurrer` solution is used to blur detected people for privacy protection.

```python
from ultralytics import solutions
import cv2

video_path = "solutions_ci_demo.mp4"
output_path = "blurred_result.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

blurrer = solutions.ObjectBlurrer(
    model="yolo26n.pt",
    classes=[0],
    blur_ratio=0.5,
    show=False
)

frame_count = 0

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    result = blurrer(frame)

    out.write(result.plot_im)

    frame_count += 1

cap.release()
out.release()

print("Object blurring completed!")
print("Frames processed:", frame_count)
print("Output saved as:", output_path)
```

Example output:

```text
Object blurring completed!
Frames processed: ...
Output saved as: blurred_result.mp4
```

The resulting video provides a privacy-preserving version of the processed footage.

### 6.4 Additional Video Analytics

Ultralytics `Analytics` is used to perform additional analytics on the video stream.

The video is processed frame by frame using OpenCV and the processed frames are written to an output video.

Example output:

```text
analytics_result.mp4
```

---

## 7. Model Evaluation

The pretrained YOLO model was evaluated using the Ultralytics validation pipeline.

The validation was performed using the COCO8 dataset as a baseline evaluation dataset.

### 7.1 Validation Code

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

validation_results = model.val(
    data="coco8.yaml",
    imgsz=640,
    batch=8
)

print("Validation completed successfully!")
```

The validation run completed successfully and produced concrete evaluation metrics.

### 7.2 Validation Results

| Metric | Result |
|---|---:|
| Precision | 0.849 |
| Recall | 0.654 |
| mAP50 | 0.906 |
| mAP50-95 | 0.665 |

### 7.3 Evaluation Metrics

#### Precision

Precision measures how many of the model's predicted detections are correct.

The model achieved:

```text
0.849
```

This indicates that a high proportion of the predicted detections were correct.

#### Recall

Recall measures how many relevant objects were successfully detected.

The model achieved:

```text
0.654
```

The lower recall compared with precision indicates that some relevant objects were missed.

#### mAP50

mAP50 measures mean Average Precision using an IoU threshold of 0.50.

The model achieved:

```text
0.906
```

This indicates strong detection performance under the 0.50 IoU evaluation condition.

#### mAP50-95

mAP50-95 measures mean Average Precision across IoU thresholds from 0.50 to 0.95.

The model achieved:

```text
0.665
```

This provides a stricter measure of object localization quality across multiple IoU thresholds.

### 7.4 Evaluation Interpretation

The validation results provide a quantitative baseline for the pretrained YOLO model.

The relatively high precision indicates that the model produced a high proportion of correct detections.

The lower recall indicates that some objects were not detected, resulting in false negatives.

Possible causes include:

- Small objects.
- Partial occlusion.
- Crowded scenes.
- Difficult lighting conditions.
- Objects blending with the background.

False positives may occur when background regions or visually similar objects are incorrectly classified as target objects.

### 7.5 IoU and Confidence Thresholds

IoU (Intersection over Union) measures the overlap between predicted and ground-truth bounding boxes.

The mAP50 metric evaluates detections at an IoU threshold of 0.50, while mAP50-95 evaluates the model across multiple IoU thresholds.

Confidence thresholds determine how confident the model must be before a detection is accepted.

The validation results are used as a baseline before custom training on a task-specific dataset.

---

## 8. Project Outputs

The project generates visual and video outputs during execution.

### Image Outputs

```text
detection_test_frame.jpg
detection_result.jpg
segmentation_result.jpg
```

### Video Outputs

```text
people_counting_result.mp4
heatmap_result.mp4
blurred_result.mp4
analytics_result.mp4
```

These outputs provide visual evidence of the executed computer vision pipeline.

---

## 9. Notebook

The complete implementation is available in:

```text
Smart_Retail_Analytics_Capstone.ipynb
```

The notebook contains:

- Project setup
- YOLO model loading
- Object detection
- Instance segmentation
- Video analytics
- Generated outputs
- Model validation
- Evaluation metrics
- Captured execution results

The notebook is intended to be executed with captured outputs so that the implementation provides evidence of real execution.

---

## 10. How to Run

### 10.1 Prerequisites

The project can be executed using Google Colab or a compatible Python environment.

### 10.2 Install Ultralytics

```python
!pip install -q ultralytics
```

### 10.3 Import Required Libraries

```python
from ultralytics import YOLO, solutions

import cv2
import os
import pandas as pd
```

### 10.4 Run the Notebook

1. Open `Smart_Retail_Analytics_Capstone.ipynb`.
2. Install the required dependencies.
3. Load the required YOLO model weights.
4. Provide the required input video or image.
5. Execute the notebook cells in order.
6. Review the generated image and video outputs.
7. Review the validation metrics.

---

## 11. Project Pipeline

```text
                    Smart Retail Analytics
                              |
                              v
                    Input Image / Video
                              |
              +---------------+---------------+
              |                               |
              v                               v
       YOLO Detection                  YOLO Segmentation
              |                               |
              v                               v
      Detection Result                Segmentation Result
              |
              v
        Video Analytics
              |
       +------+--------+-------------+
       |      |        |             |
       v      v        v             v
   Counting Heatmap  Blurring    Analytics
       |      |        |             |
       +------+--------+-------------+
                      |
                      v
                 Output Videos
                      |
                      v
                Model Evaluation
                      |
                      v
                Performance Metrics
```

---

## 12. Project Structure

```text
Smart-Retail-Analytics/
|
+-- Smart_Retail_Analytics_Capstone.ipynb
+-- README.md
+-- .gitignore
+-- .gitattributes
```

Generated files, runtime directories, datasets, model weights, and other large generated artifacts should be excluded from version control where appropriate.

---

## 13. Future Work

### 13.1 Custom Model Training

A custom dataset will be selected from a suitable source such as:

- Roboflow Universe
- Kaggle
- Open Images
- Custom collected images

The YOLO model will then be fine-tuned using `model.train()`.

Training configuration will include parameters such as:

- Number of epochs
- Image size
- Data augmentation
- Model freezing where appropriate

Training results will be analyzed for possible overfitting or underfitting.

### 13.2 Model Export

After custom training, the trained model will be exported to an optimized deployment format using `model.export()`.

Possible deployment formats include:

- ONNX
- OpenVINO
- TensorRT
- TFLite
- TorchScript

### 13.3 Application Deployment

A small application may be developed using:

- Streamlit
- Gradio
- FastAPI

The selected deployment target will be documented and justified.

---

## 14. Capstone Deliverables Status

| Deliverable | Points | Status |
|---|---:|---|
| Core Vision Tasks & Inference | 25 | Completed |
| Real-World Solution & Video Analytics | 25 | Completed |
| Model Evaluation | 25 | Completed |
| Custom Data & Training | 15 | Planned |
| Deployment & Export | 5 | Planned |
| Documentation & Evidence of Execution | 5 | In Progress |

The first three deliverables have been implemented with executed YOLO/API calls and captured outputs.

Custom training and deployment/export are planned extensions.

---

## 15. Training Program

### Training Information

**Trainee:** Saja Aljamal

**Training Program:** Computer Vision for Developers with Ultralytics

**Provider:** SDAIA Academy

**Delivery:** Learning Space

**Program Format:** 5-day capstone, on-site | 30 training hours

**Cohort / Session Dates:** August 9–13, 2026

This project was developed as part of the Computer Vision for Developers with Ultralytics training program delivered by SDAIA Academy via Learning Space.

### SDAIA Academy

GitHub organization:

https://github.com/SDAIAAcademy

---

## 16. Author

**Saja Aljamal**

Smart Retail Analytics

Computer Vision for Developers with Ultralytics Capstone