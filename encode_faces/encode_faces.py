import os
import cv2
import pickle
import numpy as np
import logging
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
from PIL import Image

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

# Load models
face_detector = MTCNN()
face_encoder = FaceNet()

# Constants
PEOPLE_DIR = "../data/face_dataset"
ENCODINGS_PATH = "../models/facenet_encodings.pkl"
REQUIRED_SIZE = (160, 160)
RECOGNITION_THRESHOLD = 0.3

# Normalizer for face embeddings
l2_normalizer = Normalizer('l2')


def get_face(img, bbox):
    """Extracts a face from an image using bounding box coordinates."""
    x1, y1, width, height = bbox
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    return img[y1:y2, x1:x2], (x1, y1), (x2, y2)


def get_encode(face_encoder, face):
    """Generates an embedding vector from a face image."""
    face = cv2.resize(face, REQUIRED_SIZE)
    face = face.astype('float32')
    return face_encoder.embeddings(np.expand_dims(face, axis=0))[0]


def load_pickle(path):
    """Loads face encodings from a pickle file."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_pickle(path, obj):
    """Saves face encodings to a pickle file."""
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def preprocess_image(img_path):
    """Reads and converts an image to RGB format."""
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def train_face_encodings():
    """Encodes faces from dataset and saves them in a pickle file."""
    encoding_dict = {}

    if not os.path.exists(PEOPLE_DIR):
        logging.error(f"Directory {PEOPLE_DIR} not found!")
        return

    for person_name in os.listdir(PEOPLE_DIR):
        person_dir = os.path.join(PEOPLE_DIR, person_name)
        encodes = []

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img_rgb = preprocess_image(img_path)

            results = face_detector.detect_faces(img_rgb)
            if results:
                res = max(results, key=lambda b: b['box'][2] * b['box'][3])  # Get largest detected face
                face, _, _ = get_face(img_rgb, res['box'])
                encodes.append(get_encode(face_encoder, face))

        if encodes:
            encoding_dict[person_name] = np.mean(encodes, axis=0)  # Average embedding vectors

    save_pickle(ENCODINGS_PATH, encoding_dict)
    logging.info(f"Encodings saved to {ENCODINGS_PATH}")


def recognize_faces(image_path):
    """Recognizes faces in a given image."""
    encoding_dict = load_pickle(ENCODINGS_PATH)
    img_rgb = preprocess_image(image_path)
    results = face_detector.detect_faces(img_rgb)

    for res in results:
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(face_encoder, face)

        name = "unknown"
        min_distance = float("inf")

        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < RECOGNITION_THRESHOLD and dist < min_distance:
                name, min_distance = db_name, dist

        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(img_rgb, pt_1, pt_2, color, 2)
        cv2.putText(img_rgb, f"{name}__{min_distance:.2f}", pt_1, cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    save_and_display_result(img_rgb)


def save_and_display_result(img_rgb):
    """Saves and displays the result image."""
    result_path = "result.jpg"
    cv2.imwrite(result_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    pil_image = Image.fromarray(img_rgb)
    plt.imshow(pil_image)
    plt.axis("off")
    plt.show()

    logging.info(f"Result saved as {result_path}")


if __name__ == "__main__":
    # Train and save encodings
    train_face_encodings()

    # Recognize faces in a test image
    test_image_path = "../data/test.jpg"
    recognize_faces(test_image_path)
