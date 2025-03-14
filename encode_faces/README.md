# Face Encoding with FaceNet

This module is responsible for encoding and recognizing faces using FaceNet. It processes images from a structured dataset, generates facial embeddings, and stores them for future recognition.

---
## How It Works
`encode_faces.py` performs two main tasks:
1. **Face Encoding** – Detects faces in images and generates FaceNet embeddings.
2. **Face Recognition** – Matches detected faces against stored embeddings to identify individuals.

---
## Preparing Your Dataset
- Store face images in `data/face_dataset/`, organized in subfolders named after each individual.
  ```
  data/face_dataset/
  ├── Alice/
  │   ├── img1.jpg
  │   ├── img2.jpg
  ├── Bob/
  │   ├── img1.jpg
  │   ├── img2.jpg
  ```
- Each folder should contain multiple images of the same person for better encoding accuracy.

---
## Running Face Encoding
Navigate to the `encode_faces` folder and execute:
```bash
cd encode_faces
python encode_faces.py
```
- This will generate and store face embeddings in `models/facenet_encodings.pkl`.

---
## Recognizing Faces in Images
Once the encodings are generated, you can run face recognition on a test image stored as data/test.jpg, and it will detect faces and display the recognized names.

---


