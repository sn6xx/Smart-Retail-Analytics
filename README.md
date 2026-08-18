# Smart Retail Analytics

**Trainee:** Saja Aljamal

## Capstone Project — Computer Vision for Developers with Ultralytics

Smart Retail Analytics is an end-to-end computer vision application developed using **Ultralytics YOLO** and **OpenCV**.

The project focuses on detecting people in retail-like video environments, applying real-world video analytics, generating privacy-preserving outputs, evaluating model performance, training a custom YOLO model on a task-specific dataset, and exporting the trained model to ONNX for deployment.

---

## 1. Project Overview

### Problem

Retail environments generate large amounts of visual data that can be used to understand customer presence and movement.

Manually reviewing video footage is time-consuming and difficult to scale. Computer vision can automate people detection and provide useful analytics from video streams.

### Solution

The Smart Retail Analytics system uses Ultralytics YOLO and OpenCV to process images and videos.

The implemented pipeline includes:

1. YOLO object detection.
2. Instance segmentation.
3. Real-world video analytics.
4. People counting.
5. Heatmap generation.
6. Privacy-preserving object blurring.
7. Additional video analytics.
8. Pretrained model evaluation.
9. Custom dataset training.
10. Custom model inference on unseen images.
11. Custom model inference on an unseen video.
12. ONNX model export.

The project demonstrates an end-to-end computer vision workflow rather than a simple object-detection demonstration.

---

## 2. Project Objectives

The main objectives of this capstone project are:

* Perform object detection using Ultralytics YOLO.
* Perform a computer vision task beyond plain detection using instance segmentation.
* Apply YOLO to a real-world video scenario.
* Analyze people movement in video.
* Count detected people.
* Generate movement heatmaps.
* Apply privacy-preserving object blurring.
* Perform additional video analytics.
* Evaluate model performance using quantitative metrics.
* Train a custom YOLO model using a task-specific dataset.
* Run the custom model on unseen images.
* Run the custom model on an unseen video.
* Export the trained model to ONNX.
* Capture and document real execution results.

---

## 3. Technologies

* Python
* Ultralytics YOLO
* OpenCV
* NumPy
* Pandas
* PyTorch
* Roboflow Universe
* Google Colab
* Google Drive
* GitHub

---

## 4. Computer Vision Models

### 4.1 Pretrained Detection Model

The project uses the pretrained YOLO model:

```text
yolo26n.pt
```

The model is used for object detection on images and video frames.

The detection model generates:

* Bounding boxes
* Class labels
* Confidence scores

### 4.2 Instance Segmentation Model

The project also uses the task-specific segmentation model:

```text
yolo26n-seg.pt
```

Instance segmentation extends object detection by generating object masks in addition to bounding boxes and class predictions.

This demonstrates a computer vision task beyond basic object detection.

---

# 5. Core Vision Tasks & Inference

## 5.1 YOLO Object Detection

A frame was extracted from the input video using OpenCV and processed using the pretrained YOLO model.

Input video:

```text
solutions_ci_demo.mp4
```

The extracted frame was saved as:

```text
detection_test_frame.jpg
```

The detection result was saved as:

```text
detection_result.jpg
```

The executed inference successfully detected people in the test frame and generated an annotated image containing bounding boxes, class labels, and confidence information.

## 5.2 Instance Segmentation

Instance segmentation was performed using the task-specific model:

```text
yolo26n-seg.pt
```

The segmentation model generates object masks in addition to bounding boxes and class predictions.

Example output:

```text
segmentation_result.jpg
```

This satisfies the requirement for a computer vision task beyond plain object detection.

---

# 6. Real-World Solution & Video Analytics

The project applies Ultralytics Solutions to a real video-processing pipeline.

The general workflow is:

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

The video-processing pipeline performs real frame-by-frame processing using OpenCV.

## 6.1 People Counting

People-counting functionality was applied to the video stream to detect and count people.

Output:

```text
people_counting_result.mp4
```

The video was processed frame by frame and the resulting annotated frames were written to an output video.

## 6.2 Heatmap Generation

Ultralytics Heatmap functionality was used to visualize movement within the video.

The heatmap provides a visual representation of areas where detected people move through the scene.

Output:

```text
heatmap_result.mp4
```

## 6.3 Privacy-Preserving Object Blurring

The Ultralytics `ObjectBlurrer` solution was used to blur detected people for privacy protection.

Output:

```text
blurred_result.mp4
```

The resulting video provides a privacy-preserving version of the processed footage.

## 6.4 Additional Video Analytics

Additional Ultralytics analytics functionality was applied to the video stream.

Output:

```text
analytics_result.mp4
```

These tasks demonstrate the use of YOLO within a real OpenCV capture, processing, and output pipeline.

---

# 7. Model Evaluation

## 7.1 Pretrained Model Evaluation

The pretrained YOLO model was evaluated using the Ultralytics validation pipeline with the COCO8 dataset as a baseline evaluation dataset.

The validation produced the following metrics:

| Metric    | Result |
| --------- | -----: |
| Precision |  0.849 |
| Recall    |  0.654 |
| mAP50     |  0.906 |
| mAP50-95  |  0.665 |

### Precision

Precision measures the proportion of predicted detections that are correct.

The model achieved:

```text
0.849
```

This indicates that a high proportion of the model's predicted detections were correct.

### Recall

Recall measures the proportion of relevant objects that were successfully detected.

The model achieved:

```text
0.654
```

The lower recall compared with precision indicates that some relevant objects were missed.

### mAP50

The model achieved:

```text
0.906
```

mAP50 measures mean Average Precision using an IoU threshold of 0.50.

### mAP50-95

The model achieved:

```text
0.665
```

mAP50-95 provides a stricter evaluation across IoU thresholds from 0.50 to 0.95.

## 7.2 Evaluation Interpretation

The validation results provide a quantitative baseline for the pretrained model.

The relatively high precision indicates that the model produced a high proportion of correct detections, while the lower recall indicates that some relevant objects were missed.

Potential causes of missed detections or incorrect predictions can include:

* Small objects.
* Partial occlusion.
* Crowded scenes.
* Difficult lighting conditions.
* Objects blending with the background.

The evaluation was used as a baseline before custom training on the task-specific dataset.

## 7.3 IoU and Confidence Thresholds

IoU (Intersection over Union) measures the overlap between predicted and ground-truth bounding boxes.

The project uses confidence thresholds during inference to control which detections are accepted.

For the custom model inference stages, a confidence threshold of:

```text
0.5
```

was used.

---

# 8. Custom Data & Training

## 8.1 Dataset Source

The custom training dataset was obtained from **Roboflow Universe**.

**Dataset:** People Detection - Thermal

**Task:** Object Detection

**Class:** person

**Source:** Roboflow Universe

**License:** CC BY 4.0

The dataset is publicly available through Roboflow Universe and provides YOLO-compatible dataset exports.

Dataset source:

[People Detection - Thermal — Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/people-detection-thermal?utm_source=chatgpt.com)

The dataset was downloaded and prepared in YOLO format for use with Ultralytics YOLO26n.

The prepared dataset was stored locally at:

```text
/content/custom_dataset
```

## 8.2 Custom Training

The YOLO26n model was fine-tuned using the custom people-detection dataset.

The training process generated trained model weights.

The best trained model was saved at:

```text
/content/runs/detect/train/weights/best.pt
```

The custom training stage demonstrates fine-tuning on a task-specific dataset rather than re-running the COCO8 warm-up demonstration.

---

# 9. Custom Model Inference

## 9.1 Custom Model Image Inference

The trained YOLO26n model was loaded from:

```text
/content/runs/detect/train/weights/best.pt
```

The model was used to perform inference on unseen images from the custom dataset.

The prediction results were saved by Ultralytics under:

```text
/content/runs/detect/predict/
```

The results were visually inspected by displaying individual prediction images and multiple randomly selected prediction images.

The prediction results showed successful people detection with bounding boxes.

## 9.2 Custom Model Video Inference

The trained custom model was also evaluated on an unseen video:

```text
test_video.mp4
```

The model was loaded using:

```python
from ultralytics import YOLO

model = YOLO("/content/runs/detect/train/weights/best.pt")

results = model.predict(
    source="/content/test_video.mp4",
    save=True,
    conf=0.5
)

print("Video inference completed successfully.")
```

The inference completed successfully and generated an annotated video.

Ultralytics initially generated the output in AVI format:

```text
test_video.avi
```

The generated video was subsequently converted to MP4 using FFmpeg for easier playback and visualization.

This demonstrates that the custom-trained model was applied to an unseen video rather than only to training images.

---

# 10. Deployment & Export

## 10.1 ONNX Export

The trained custom model was exported to ONNX format using the Ultralytics Python API.

```python
from ultralytics import YOLO

model = YOLO("/content/runs/detect/train/weights/best.pt")

export_path = model.export(format="onnx")

print("Model export completed successfully!")
print("Exported model:", export_path)
```

The export completed successfully.

Output:

```text
/content/runs/detect/train/weights/best.onnx
```

The exported model was verified after the export.

Verification result:

```text
ONNX model exists: True
ONNX model path: /content/runs/detect/train/weights/best.onnx
ONNX model size: 9.35 MB
```

The ONNX model provides a portable format suitable for deployment and inference outside the original PyTorch training environment.

---

# 11. Project Outputs

## Image Outputs

```text
detection_test_frame.jpg
detection_result.jpg
segmentation_result.jpg
```

Custom model prediction images are generated under:

```text
runs/detect/predict/
```

## Video Outputs

```text
people_counting_result.mp4
heatmap_result.mp4
blurred_result.mp4
analytics_result.mp4
test_video.mp4
```

The custom model video inference output was generated under the corresponding Ultralytics prediction directory.

## Model Outputs

```text
best.pt
best.onnx
```

The trained model files are stored under:

```text
runs/detect/train/weights/
```

---

# 12. Project Pipeline

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
        Real-World Video
            Analytics
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
               Custom Training
                      |
                      v
             Custom Model Inference
                 |          |
                 v          v
               Image      Video
                            |
                            v
                       ONNX Export
```

---

# 13. Project Structure

The GitHub repository contains the main notebook and project documentation:

```text
Smart-Retail-Analytics/
|
+-- Smart_Retail_Analytics_Capstone.ipynb
+-- README.md
+-- .gitignore
+-- .gitattributes
```

Large generated files, datasets, model weights, and runtime artifacts are excluded from Git version control where appropriate.

Examples include:

```text
runs/
datasets/
custom_dataset/
*.pt
*.onnx
*.mp4
*.avi
```

The executed project files and generated artifacts are maintained separately through the project backup workflow.

---

# 14. How to Run

## 14.1 Prerequisites

The project can be executed using Google Colab or a compatible Python environment.

Recommended environment:

* Python 3.12+
* PyTorch
* Ultralytics
* OpenCV
* NumPy
* Pandas

## 14.2 Install Ultralytics

```python
!pip install -q ultralytics
```

## 14.3 Import Required Libraries

```python
from ultralytics import YOLO, solutions

import cv2
import os
import pandas as pd
```

## 14.4 Run the Notebook

1. Open `Smart_Retail_Analytics_Capstone.ipynb`.
2. Install the required dependencies.
3. Load the required YOLO models.
4. Provide the required input image or video.
5. Execute the notebook cells in order.
6. Review the detection and segmentation outputs.
7. Review the real-world video analytics outputs.
8. Run model validation.
9. Prepare the custom dataset from Roboflow Universe.
10. Train the custom model.
11. Run custom model inference.
12. Export the trained model to ONNX.

---

# 15. Evidence of Execution

The notebook contains captured execution results from the implemented computer vision pipeline.

Evidence includes:

* Successful YOLO model loading.
* Successful object detection.
* Detection output images.
* Instance segmentation output.
* People-counting video.
* Heatmap video.
* Privacy-preserving blurred video.
* Additional analytics video.
* Pretrained model validation metrics.
* Custom dataset training.
* Custom model prediction images.
* Custom model video inference.
* Successful ONNX export.
* ONNX model verification.

The executed project was also backed up to Google Drive to preserve the trained model, dataset, generated outputs, and project runtime files.

---

# 16. Capstone Deliverables Status

| Deliverable                           | Points | Status    |
| ------------------------------------- | -----: | --------- |
| Core Vision Tasks & Inference         |     25 | Completed |
| Real-World Solution & Video Analytics |     25 | Completed |
| Model Evaluation                      |     25 | Completed |
| Custom Data & Training                |     15 | Completed |
| Deployment & Export                   |      5 | Completed |
| Documentation & Evidence of Execution |      5 | Completed |

The project implements all six capstone deliverables using executed Ultralytics API calls and captured execution evidence.

---

# 17. Training Program

## Training Information

**Trainee:** Saja Aljamal

**Training Program:** Computer Vision for Developers with Ultralytics

**Provider:** SDAIA Academy

**Delivery:** Learning Space

**Program Format:** 5-day capstone, on-site | 30 training hours

**Cohort / Session Dates:** August 9–13, 2026

This project was developed as part of the **Computer Vision for Developers with Ultralytics** training program delivered by **SDAIA Academy** via Learning Space.

## SDAIA Academy

GitHub organization:

https://github.com/SDAIAAcademy

---

# 18. Author

**Saja Aljamal**

**Project:** Smart Retail Analytics

**Program:** Computer Vision for Developers with Ultralytics

**Provider:** SDAIA Academy
