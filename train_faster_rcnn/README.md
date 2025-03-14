# Training Faster R-CNN for Object Detection

This module trains a **Faster R-CNN** model for detecting objects (people and dogs) using a subset of the **Pascal VOC 2012** dataset.

---
## How It Works
`faster_rcnn_training_gpt.ipynb` performs the following steps using a custom dataset. The CSV files (`data/train_Pascal_custom.csv` and `data/test_Pascal_custom.csv`) were manually created and contain a random subset of images with people and dogs, allowing for faster training and testing purposes.
1. **Loads the Pascal VOC dataset** – Uses a custom CSV file (`data/train_Pascal_custom.csv`) with bounding box annotations.
2. **Prepares the dataset** – Organizes images and labels for training.
3. **Trains Faster R-CNN** – Fine-tunes a pre-trained Faster R-CNN model.
4. **Evaluates the model** – Runs inference on test images.
5. **Saves the trained model** – Stores the model weights as `models/detection_model.pth`.

---
## Dataset Preparation
- Ensure that `data/VOC2012_images/` contains the Pascal VOC images.
- The dataset annotations are stored in:
  - `data/train_Pascal_custom.csv` (Training set)
  - `data/test_Pascal_custom.csv` (Test set)



---
## Customization
- **Modify Classes:** You can train the model to detect additional object categories beyond `people` and `dog` by modifying the code and updating the CSV files.
- **Train on More Data:** Expand the training dataset by adding more images to `data/VOC2012_images/` and updating the CSV files.

---

