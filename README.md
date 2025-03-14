# Object-Face-Detection-Recognition

## Overview
This project combines Faster R-CNN, MTCNN, and FaceNet to perform multi-stage object detection and face recognition. The pipeline follows these steps:
1. **Faster R-CNN** detects objects (people and dogs) in an image.
2. **MTCNN** runs only on regions classified as 'people' to detect faces.
3. **FaceNet** encodes and recognizes detected faces against a database of known faces.

## Repository Structure
```
📂 Object-Face-Detection-Recognition
│── 📂 train_faster_rcnn/                   # Folder for training Faster R-CNN
│   ├── train_faster_rcnn.ipynb             # Jupyter Notebook for training
│   ├── README.md                           
│
│── 📂 hierarchical_detection/              # Full pipeline
│   ├── multi_stage_detection.ipynb         # Jupyter Notebook for the full pipeline
│   ├── README.md                           
│
│── 📂 encode_faces/                        # Encoding faces using FaceNet
│   ├── encode_faces.py                     # Python script for encoding faces
│   ├── README.md                           
│
│── 📂 models/                              # Trained models
│   ├── fasterrcnn_people_dog_trained.pth   # Trained Faster R-CNN model
│   ├── facenet_encodings.pkl               # Trained FaceNet embeddings
│
│── 📂 data/                                # Dataset and annotation files
│   ├── 📂 VOC2012_images/                  # Contains only images from Pascal VOC 2012 dataset
│   ├── train_Pascal_custom.csv             # Custom annotation file for training 
│   ├── test_Pascal_custom.csv              # Custom annotation file for testing 
│   ├── test.jpg                            # Example test image
│   ├── 📂 face_dataset/                    # Stores images for FaceNet encoding
│   │   ├── 📂 Alice/                       # Folder for person "Alice"
│   │   │   ├── img1.jpg                    # Sample image
│   │   │   ├── img2.jpg                    # Sample image
│   │   ├── 📂 Bob/                         # Folder for person "Bob"
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   ├── …                               # More persons
│
│── requirements.txt                        
│── README.md                               
```

## Features
- **Object Detection**: Identifies people and dogs using a fine-tuned Faster R-CNN model.
- **Face Detection**: Uses MTCNN to extract faces from detected 'person' objects.
- **Face Recognition**: Uses FaceNet embeddings to recognize known individuals.
- **Efficient Pipeline**: Runs MTCNN and FaceNet only on relevant regions to improve speed.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ariobarzanes98/FasterRCNN-MTCNN-FaceNet.git
   cd Object-Face-Detection-Recognition
   ```
2. Create and activate a Conda environment with Python 3.9:
   ```bash
   conda create --name face_detection_env python=3.9 -y
   conda activate face_detection_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the Pascal VOC 2012 dataset (VOCtrainval_11-May-2012) and place it inside the `data/` folder.

## Usage

- [Train Faster R-CNN](train_faster_rcnn/) – Set up and train the object detection model.
- [Encode Faces with FaceNet](encode_faces/) – Process and store face embeddings for recognition.
- [Run the Full Detection Pipeline](hierarchical_detection/) – Perform object detection with Faster R-CNN, apply face detection using MTCNN on detected people, and recognize faces with FaceNet in a sequential pipeline.
