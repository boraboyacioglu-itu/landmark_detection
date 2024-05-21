from typing import List, Tuple
from _dlib_pybind11 import fhog_object_detector as dlib_pybind11_fhog_object_detector
from _dlib_pybind11 import shape_predictor as dlib_pybind11_shape_predictor

import cv2
import mediapipe as mp
import numpy as np

from imutils import face_utils

from plotly import graph_objects as go
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

def preprocess(image, target_size) -> np.ndarray:
    """ Preprocess an image. """
    
    # Convert to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image.
    image = cv2.resize(image, target_size)
    
    return image

def extract_landmarks(detector, image) -> np.ndarray:
    """ Extract and normalize the landmarks from the image. """
    
    # Convert the image to MediaPipe image.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
    # Detect the landmarks.
    face_landmarks = detector.detect(mp_image).face_landmarks
    
    # Define the the landmark extractor.
    coords = lambda lm: [lm.x, lm.y, lm.z]
    
    if face_landmarks and len(face_landmarks) != 468:
        face_landmarks = face_landmarks[0]
            
    # Extract the landmarks.
    lm = [
        coords(normalized_landmark)
        for normalized_landmark in face_landmarks
    ]
    
    if len(lm) != 478:
        lm = np.zeros((478, 3))
    
    return lm

def train_test_split(X1: List[np.ndarray], X2: List[np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Split the data into train and test sets. """
    
    # Initialize the train and test sets.
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = [], [], [], [], [], []
    
    # Initialize the ranges.
    ranges = range(len(X1))
    
    # Indexes with only 1 photo.
    C0_ind = [i for i, vals in enumerate(X1) if len(vals) == 1]
    
    # Indexes with more than 1 and less than 6 photos.
    C1_ind = [i for i, vals in enumerate(X1) if len(vals) < 6 and i not in C0_ind]
    
    # Indexes with at least 6 photos.
    C2_ind = [i for i, vals in enumerate(X1) if len(vals) >= 6]
    
    for i in ranges:
        # If the index is in C0,
        # append the values to the train set.
        if i in C0_ind:
            X1_train.append(X1[i])
            X2_train.append(X2[i])
            y_train.append(y[i])
            
            continue
        
        # If the index is in C1,
        # append all values except the last one to the train set,
        # and the last value to the test set.
        if i in C1_ind:
            # Train set.
            X1_train.extend(X1[i][:-1])
            X2_train.extend(X2[i][:-1])
            y_train.extend([y[i]] * (len(X1[i]) - 1))
            
            # Test set.
            X1_test.append(X1[i][-1])
            X2_test.append(X2[i][-1])
            y_test.append(y[i])
            
            continue
        
        # If the index is in C2,
        # append 2/3 of the values to the train set,
        # and 1/3 of the values to the test set.
        if i in C2_ind:
            split = 2 * len(X1[i]) // 3
            
            # Train set.
            X1_train.extend(X1[i][:split])
            X2_train.extend(X2[i][:split])
            y_train.extend([y[i]] * split)
            
            # Test set.
            X1_test.extend(X1[i][split:])
            X2_test.extend(X2[i][split:])
            y_test.extend([y[i]] * (len(X1[i]) - split))
            
            continue
        
    return X1_train, X1_test, X2_train, X2_test, y_train, y_test

def stabilized_lm(lm: np.ndarray, N: int) -> np.ndarray:
    """ Stabilized landmarks. """
    
    # Std dev of x, y, z coordinates for each landmark.
    lm_var = np.std(lm, axis=0)

    # Total variability of each landmark.
    total_var = np.sum(lm_var, axis=1)

    # Indexes of the most stable landmarks.
    lm_stable = np.argsort(total_var)[:N]
    
    return lm_stable

def realign_lm(lm: np.ndarray, lm_stable: np.ndarray, ref: int = 0) -> np.ndarray:
    """ Realign the landmarks. """
    
    # Extract the stable landmarks.
    ref_points = lm[ref, lm_stable]
    face_points = lm[:, lm_stable]
    
    # Define the align function.
    def align(ref_points, face_points, point):
        # Create the aligned points.
        aligned_points_s = procrustes(ref_points, face_points)[1]
        
        # Calculate the rotation and scale.
        R, scale = orthogonal_procrustes(face_points, aligned_points_s)
        
        # Do the alignment transformations.
        point -= np.mean(face_points, axis=0)
        point = point @ R.T * scale
        point += np.mean(ref_points, axis=0)
        return point
        
    # Align the landmarks.
    aligned_lm = [
        align(ref_points, face_points[i], point)
        if np.any(point)
        else point
        for i, point in enumerate(lm)
    ]
    
    return np.array(aligned_lm)

def extract_2d_lm(imgs: np.ndarray, detector: dlib_pybind11_fhog_object_detector, predictor: dlib_pybind11_shape_predictor) -> np.ndarray:
    lms = []
    for i, img in enumerate(imgs):
        print(f"\rExtracting landmarks... {100 * (i + 1) / len(imgs):.2f}%", end="")
        
        # Convert the image to grayscale.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces.
        faces = detector(img, 1)
        
        if len(faces) == 0:
            lms.append(np.zeros((68, 2)))
            continue
        
        # Predict the landmarks.
        lm = predictor(img, faces[0])
        lm = face_utils.shape_to_np(lm)
        
        # Append the landmarks.
        lms.append(lm)
    print("")
    
    return np.array(lms)

def realign_img(lm2d: np.ndarray, imgs: np.ndarray, lm_stable: np.ndarray, ref: int = 0) -> np.ndarray:
    """ Realign the image. """
    
    # Extract the stable landmarks.
    ref_points = lm2d[ref]
    
    # Create the aligned images.
    aligned_imgs = np.zeros_like(imgs)
    
    for i, lm in enumerate(lm2d):
        print(f"\rRealigning images... {100 * (i + 1) / len(lm2d):.2f}%", end="")
        
        # Compute the transformation matrix.
        transformation_matrix = cv2.estimateAffinePartial2D(
            ref_points, lm, method=cv2.RANSAC
        )[0]
        
        # Apply the transformation to the image.
        aligned_imgs[i] = cv2.warpAffine(
            imgs[i], transformation_matrix, (imgs[i].shape[1], imgs[i].shape[0])
        )
    print("")
    
    return aligned_imgs

def plot_lm(lm: np.ndarray, stable_indices=None) -> None:
    """ Plot the landmarks. """
    
    # Initialise the colors.
    colors = np.array(lm[:, 2], copy=True)
    colorscale = 'Viridis'
    
    # Change the colours based on the stable indices.
    if stable_indices is not None:
        # Change all the colours to 0.
        colors[:] = 0
        
        # Change the colours of the stable landmarks to 1.
        colors[stable_indices] = 1
        
        # Change the colorscale.
        colorscale = [[0, 'black'], [1, 'red']]
    
    # Create a scatter plot of the 3D points.
    fig = go.Figure(data=[go.Scatter3d(
        x=lm[:, 0],
        y=lm[:, 1],
        z=lm[:, 2],
        mode='markers',
        marker=dict(size=2, color=colors, colorscale=colorscale, opacity=0.8)
    )])

    # Update the layout.
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
        margin={'l': 10, 'r': 10, 'b': 10, 't': 10}
    )

    # Show the plot.
    fig.show()