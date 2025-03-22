import cv2
import dlib
import numpy as np
import os
import joblib
from skimage import io, transform
from imutils import face_utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import logging
import random
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def align_face(image, landmarks, desired_face_width=256, desired_eye_dist=0.6):
    """
    Aligns and crops a face based on facial landmarks.
    """
    try:
        # Get the coordinates of the left and right eyes
        left_eye = np.mean(landmarks[36:42], axis=0).astype(int)
        right_eye = np.mean(landmarks[42:48], axis=0).astype(int)

        # Calculate the angle between the eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate the center point between the eyes and ensure it's a tuple of integers
        eye_center = tuple(map(int, (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )))

        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)

        # Rotate the image
        (h, w) = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # Recalculate the eye coordinates after rotation
        left_eye_rot = np.array(
            [
                M[0, 0] * left_eye[0] + M[0, 1] * left_eye[1] + M[0, 2],
                M[1, 0] * left_eye[0] + M[1, 1] * left_eye[1] + M[1, 2],
            ]
        )
        right_eye_rot = np.array(
            [
                M[0, 0] * right_eye[0] + M[0, 1] * right_eye[1] + M[0, 2],
                M[1, 0] * right_eye[0] + M[1, 1] * right_eye[1] + M[1, 2],
            ]
        )

        # Calculate the scale
        dist = np.linalg.norm(right_eye_rot - left_eye_rot)
        desired_dist = desired_face_width * desired_eye_dist
        scale = desired_dist / dist

        # Calculate the translation
        eyes_center = (left_eye_rot + right_eye_rot) / 2
        offsetX = desired_face_width * 0.5 - eyes_center[0]
        offsetY = desired_face_width * 0.4 - eyes_center[1]

        # Create the affine transformation matrix
        affine_matrix = np.array(
            [[scale, 0, offsetX], [0, scale, offsetY]], dtype=np.float32
        )

        # Apply the affine transformation
        aligned_face = cv2.warpAffine(
            rotated,
            affine_matrix,
            (desired_face_width, desired_face_width),
            flags=cv2.INTER_CUBIC,
        )

        return aligned_face
    except Exception as e:
        logger.error(f"Error in face alignment: {e}")
        return None


def extract_facial_landmarks(
    image_path, predictor, detector, desired_face_width=256, return_image=False
):
    """
    Extracts facial landmarks from an image, including normalization steps.
    Returns landmarks and optionally the aligned face image.
    """
    try:
        img = io.imread(image_path)
        if img.shape[2] == 4:  # Handle RGBA images
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            return None if not return_image else (None, None)

        # Find the largest face
        if len(faces) > 1:
            face_areas = [(face.right() - face.left()) * (face.bottom() - face.top()) for face in faces]
            face = faces[np.argmax(face_areas)]
        else:
            face = faces[0]

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Align the face
        aligned_face = align_face(img, landmarks, desired_face_width)
        if aligned_face is None:
            return None if not return_image else (None, None)

        # Convert the aligned face to grayscale
        aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)

        # Histogram equalization
        aligned_gray = cv2.equalizeHist(aligned_gray)

        # Re-detect landmarks on the aligned face
        face_rect = dlib.rectangle(0, 0, desired_face_width, desired_face_width)
        aligned_landmarks = predictor(aligned_gray, face_rect)
        aligned_landmarks = face_utils.shape_to_np(aligned_landmarks)

        if return_image:
            return aligned_landmarks.flatten(), aligned_face
        return aligned_landmarks.flatten()

    except Exception as e:
        logger.error(f"Error extracting landmarks from {image_path}: {e}")
        return None if not return_image else (None, None)


def augment_image(image, landmarks, max_rotation=10, max_shift=10, max_scale=0.1):
    """
    Apply random augmentation to an image and update landmarks accordingly.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Random rotation
    angle = random.uniform(-max_rotation, max_rotation)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Random scale
    scale = random.uniform(1.0 - max_scale, 1.0 + max_scale)
    M_rot[0, 0] *= scale
    M_rot[1, 1] *= scale

    # Random shift
    tx = random.uniform(-max_shift, max_shift)
    ty = random.uniform(-max_shift, max_shift)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty

    # Apply transformation to image
    augmented_image = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_CUBIC)

    # Apply transformation to landmarks
    landmarks_reshaped = landmarks.reshape(-1, 2)
    augmented_landmarks = np.zeros_like(landmarks_reshaped)

    for i, (x, y) in enumerate(landmarks_reshaped):
        augmented_landmarks[i, 0] = M_rot[0, 0] * x + M_rot[0, 1] * y + M_rot[0, 2]
        augmented_landmarks[i, 1] = M_rot[1, 0] * x + M_rot[1, 1] * y + M_rot[1, 2]

    return augmented_image, augmented_landmarks.flatten()


def create_training_dataset(
    data_dir, predictor_path, augmentation=True, augmentation_factor=2
):
    """
    Creates a training dataset by extracting facial landmarks.
    Includes data augmentation if specified.
    """
    X = []
    y = []
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()

    # Get list of all race directories
    race_dirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    
    logger.info(f"Found {len(race_dirs)} race categories: {race_dirs}")

    for race_label in race_dirs:
        race_dir = os.path.join(data_dir, race_label)
        image_files = [
            f
            for f in os.listdir(race_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        logger.info(f"Processing {len(image_files)} images for race: {race_label}")
        
        for image_file in tqdm(image_files, desc=f"Processing {race_label}"):
            image_path = os.path.join(race_dir, image_file)
            
            # Extract landmarks and aligned face
            result = extract_facial_landmarks(
                image_path, predictor, detector, return_image=True
            )
            
            if result is None:
                logger.warning(f"No face detected in {image_path}")
                continue
                
            landmarks, aligned_face = result
            
            if landmarks is None or aligned_face is None:
                logger.warning(f"Failed to process {image_path}")
                continue
                
            X.append(landmarks)
            y.append(race_label)
            
            # Data augmentation
            if augmentation:
                for _ in range(augmentation_factor):
                    aug_face, aug_landmarks = augment_image(aligned_face, landmarks)
                    X.append(aug_landmarks)
                    y.append(race_label)

    logger.info(f"Created dataset with {len(X)} samples")
    return X, y


def visualize_results(model, X_test, y_test, class_names):
    """
    Visualize the confusion matrix and classification report.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = model.predict(X_test)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    
    # Print and save classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")
    
    with open("classification_report.txt", "w") as f:
        f.write(report)


def train_model(X, y):
    """
    Train the model with hyperparameter tuning.
    """
    X = np.array(X)
    y = np.array(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Create a pipeline with scaling and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    # Perform grid search
    logger.info("Starting hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Visualize results
    visualize_results(best_model, X_test, y_test, np.unique(y))
    
    return best_model, X_test, y_test


if __name__ == "__main__":
    data_dir = "training_data"  # Replace with your training data directory
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    # Check if predictor file exists
    if not os.path.exists(predictor_path):
        logger.error(
            f"Predictor file not found: {predictor_path}. "
            "Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )
        exit(1)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        exit(1)
    
    logger.info("Starting dataset creation...")
    X, y = create_training_dataset(
        data_dir, predictor_path, augmentation=True, augmentation_factor=2
    )
    
    if len(X) == 0:
        logger.error("No training data found or processed.")
        exit(1)
    
    logger.info("Starting model training...")
    model, X_test, y_test = train_model(X, y)
    
    # Save the model
    model_filename = "race_detection_model.pkl"
    joblib.dump(model, model_filename)
    logger.info(f"Trained model saved to {model_filename}")
    
    # Save model metadata
    metadata = {
        "classes": list(np.unique(y)),
        "feature_count": X[0].shape[0],
        "training_samples": len(X),
        "test_accuracy": accuracy_score(y_test, model.predict(X_test))
    }
    joblib.dump(metadata, "model_metadata.pkl")
    logger.info("Model training complete!")
