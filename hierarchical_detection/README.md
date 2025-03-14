# Hierarchical Detection and Recognition

`multi_stage_detection.ipynb` applies a full detection pipeline, identifying people and dogs with Faster R-CNN, detecting faces with MTCNN, and recognizing individuals using FaceNet.

---
## Pipeline Details
### 1️⃣ Object Detection with Faster R-CNN
- Loads a pre-trained **Faster R-CNN** model from `models/detection_model.pth`.
- Detects **people and dogs** in an input image.
- Draws bounding boxes around detected objects.

### 2️⃣ Face Detection with MTCNN (Only on People)
- Runs **MTCNN** only on regions detected as 'people' by Faster R-CNN.
- Extracts face regions from detected individuals.

### 3️⃣ Face Recognition with FaceNet
- Encodes detected faces using **FaceNet**.
- Matches faces with known individuals stored in `models/facenet_encodings.pkl`.
- Displays the recognized name and similarity score.

---
## Expected Output
- **Red boxes** → Detected objects (people & dogs).
- **Green boxes** → Detected faces within people.
- **Blue text** → Recognized name and similarity score.

---
## Customization
- **Use Your Own Face Encodings:** If you want to recognize new faces, follow the [Face Encoding Guide](../encode_faces/README.md) to generate your own embeddings.
- **Train Your Own Detection Model:** If you want to train the object detection model instead of using the provided pre-trained weights, see the [Training Guide](../train_faster_rcnn/README.md).

---

