import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from mtcnn import MTCNN
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# HOG
def hog_feature(image):
    # Convert to gray scale
    gray_image = rgb2gray(image)
    # Calculate the HOG features for the resized image
    hog_features = hog(gray_image, orientations=30, pixels_per_cell=(6, 6),
                       cells_per_block=(1, 1), block_norm='L2-Hys', transform_sqrt=True)
    return hog_features


# LBP
def lbp_feature(image):
    # Convert to gray scale
    image = rgb2gray(image)
    # Calculate the LBP features
    lbp = local_binary_pattern(image, 8, 1)

    # Histogram of LBP features
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def resize(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


# Join features
def concatenate(f1, f2):
    return np.concatenate([f1, f2])


# Crop image
def crop_image(image, x, y, width, height):
    cropped_image = image[y:y + height, x:x + width]
    return cropped_image


# Detect all faces in an image
def detect_faces(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces


# Detect lip in a face image
def detect_lip(img):
    # Detect faces in the image using a CascadeClassifier object
    faces = detect_faces(img)
    # Create an MTCNN object for detecting lips
    detector = MTCNN()

    # If any face is not detected
    if len(faces) == 0:
        return resize(img, 70)
    else:
        face_img = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
        results = detector.detect_faces(face_img)

        if results:
            left_lip = results[0]['keypoints']['mouth_left']
            right_lip = results[0]['keypoints']['mouth_right']
            # Crop lip
            crop = crop_image(face_img, left_lip[0] - 6, left_lip[1] - 7, right_lip[0] + 10, right_lip[1] + 10)
            return resize(crop, 70)
        else:
            # If the facial features are not recognized
            return resize(face_img, 70)


def detect_lips(img):
    result_lips = []
    faces = detect_faces(img)
    # Create an MTCNN object for detecting lips
    detector = MTCNN()

    # Detect lips in each face using the MTCNN object and draw a rectangle around them using cv2.rectangle()
    for (x, y, w, h) in faces:
        face_img = img[y:y + h, x:x + w]
        results = detector.detect_faces(face_img)
        if results:

            left_lip = results[0]['keypoints']['mouth_left']
            right_lip = results[0]['keypoints']['mouth_right']
            # Crop lip
            crop = crop_image(face_img, left_lip[0] - 6, left_lip[1] - 7, right_lip[0] + 10, right_lip[1] + 10)
            result_lips.append(resize(crop, 70))

        else:
            result_lips.append(resize(face_img, 70))

    if len(faces) == 0:
        result_lips.append(resize(img, 70))

    return result_lips


def detect_lip_test(img, faces, face_img):
    # Create an MTCNN object for detecting lips
    detector = MTCNN()
    # Detect lips in each face using the MTCNN object and draw a rectangle around them using cv2.rectangle()
    if len(faces) == 0:
        return resize(img, 70)
    else:
        results = detector.detect_faces(face_img)
        if results:
            left_lip = results[0]['keypoints']['mouth_left']
            right_lip = results[0]['keypoints']['mouth_right']
            crop = crop_image(face_img, left_lip[0] - 6, left_lip[1] - 7, right_lip[0] + 10, right_lip[1] + 10)
            return resize(crop, 70)

        else:
            return resize(face_img, 70)


def sift(image):
    # Convert to gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray_img, None)

    num_bins = 128

    bin_size = 1 / num_bins

    hist, _ = np.histogram(descriptors, bins=num_bins, range=(0, 1))

    hist1_norm = hist / np.sum(hist)
    return hist1_norm


def load_svm():
    with open('detect_smile_model30.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf


def save_svm(clf):
    # Save SVM model
    with open('detect_smile_model30.pkl', 'wb') as f:
        pickle.dump(clf, f)


def extract_features(path):
    arr = []
    v = 0
    for filename in os.listdir(path):
        # Check format
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image
            img_path = os.path.join(path, filename)
            image = cv2.imread(img_path)

            # Resize the image to a smaller size for faster processing
            image = resize(image, 100)
            # Detect lip in face
            img = detect_lip(image)

            hog_features = hog_feature(img)
            #  LBP
            lbp_features = lbp_feature(img)

            # Combination of HOG & LBP
            features = concatenate(hog_features, lbp_features)

            arr.append(features)
            v += 1
            print(v)

    return arr


def extract_labels(path):
    c = 0
    with open(path, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        c += 1
        labels.append(int(line.split()[0]))

    return labels
