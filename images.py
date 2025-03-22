import cv2
import dlib
import numpy as np
import os
import joblib
from skimage import io
from imutils import face_utils
import argparse
import logging
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("advanced_model.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global constant for face alignment
DESIRED_FACE_WIDTH = 256

# Mapping for horizontal flip of 68 facial landmarks based on dlib's model.
FLIP_MAP = {
    0: 16,
    1: 15,
    2: 14,
    3: 13,
    4: 12,
    5: 11,
    6: 10,
    7: 9,
    8: 8,
    9: 7,
    10: 6,
    11: 5,
    12: 4,
    13: 3,
    14: 2,
    15: 1,
    16: 0,
    17: 26,
    18: 25,
    19: 24,
    20: 23,
    21: 22,
    22: 21,
    23: 20,
    24: 19,
    25: 18,
    26: 17,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 35,
    32: 34,
    33: 33,
    34: 32,
    35: 31,
    36: 45,
    37: 44,
    38: 43,
    39: 42,
    40: 47,
    41: 46,
    42: 41,
    43: 40,
    44: 39,
    45: 38,
    46: 37,
    47: 36,
    48: 54,
    49: 53,
    50: 52,
    51: 51,
    52: 50,
    53: 49,
    54: 48,
    55: 59,
    56: 58,
    57: 57,
    58: 56,
    59: 55,
    60: 64,
    61: 63,
    62: 62,
    63: 61,
    64: 60,
    65: 67,
    66: 66,
    67: 65,
}


def align_face(image, landmarks, desired_face_width=256, desired_eye_dist=0.6):
    """
    Aligns and crops a face based on facial landmarks.
    Includes improved error handling and validation.
    """
    try:
        # Validate inputs
        if image is None or landmarks is None:
            logger.error("Invalid input: image or landmarks is None")
            return None
            
        # Create a copy of the image to avoid modification of the original
        image = image.copy()
        
        # Validate image dimensions
        if len(image.shape) != 3:
            logger.error("Invalid image format: expected 3 channels")
            return None
            
        # Convert landmarks to float32 for precise calculations
        landmarks = landmarks.astype(np.float32)
        
        # Check for NaN values in landmarks
        if np.isnan(landmarks).any():
            logger.error("Invalid landmarks: contains NaN values")
            return None
            
        # Calculate the average positions for the left and right eyes
        left_eye_pts = landmarks[36:42]
        right_eye_pts = landmarks[42:48]
        
        # Validate eye landmarks
        if len(left_eye_pts) == 0 or len(right_eye_pts) == 0:
            logger.error("Invalid eye landmarks")
            return None
            
        # Calculate eye centers with validation
        left_eye = np.mean(left_eye_pts, axis=0)
        right_eye = np.mean(right_eye_pts, axis=0)
        
        # Ensure eye coordinates are valid
        if np.isnan(left_eye).any() or np.isnan(right_eye).any():
            logger.error("Invalid eye coordinates: NaN values detected")
            return None
            
        # Convert to integer coordinates
        left_eye = left_eye.astype(np.int32)
        right_eye = right_eye.astype(np.int32)
        
        # Calculate the angle between the eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        
        # Avoid division by zero
        if dX == 0 and dY == 0:
            logger.error("Invalid eye positions: eyes are at the same position")
            return None
            
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate the desired right eye position
        desired_right_eye_x = 1.0 - desired_eye_dist
        
        # Calculate the scale based on the desired distance between eyes
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        
        # Avoid division by zero in scaling
        if dist < 1e-6:
            logger.error("Eye distance too small")
            return None
            
        desired_dist = desired_eye_dist * desired_face_width
        scale = desired_dist / dist
        
        # Validate scale factor
        if scale <= 0 or scale > 10:  # Arbitrary upper limit to prevent extreme scaling
            logger.error(f"Invalid scale factor: {scale}")
            return None
            
        # Calculate the center point between the eyes
        eyes_center = (int((left_eye[0] + right_eye[0]) // 2),
                      int((left_eye[1] + right_eye[1]) // 2))
        
        # Validate center point
        if not all(isinstance(x, (int, np.integer)) for x in eyes_center):
            logger.error("Invalid center point coordinates")
            return None
            
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update the translation component of the matrix
        tX = desired_face_width * 0.5
        tY = desired_face_width * 0.35
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply the affine transformation
        output_size = (desired_face_width, desired_face_width)
        aligned_face = cv2.warpAffine(
            image,
            M,
            output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Validate output
        if aligned_face is None or aligned_face.size == 0:
            logger.error("Failed to create aligned face")
            return None
            
        return aligned_face
        
    except Exception as e:
        logger.error(f"Error in face alignment: {str(e)}")
        return None



def extract_facial_landmarks(
    image_path, predictor, detector, desired_face_width=DESIRED_FACE_WIDTH,
    return_image=False, largest_only=False
):
    """
    Extracts facial landmarks from an image.
    For training, if largest_only=True, only the largest face is processed.
    Normalizes and aligns the face before re-detecting landmarks.
    """
    try:
        if isinstance(image_path, str):
            img = io.imread(image_path)
            if img.ndim == 3 and img.shape[2] == 4:  # RGBA images
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = image_path.copy()  # Always work with a copy

        # Make sure we have a color image
        if img.ndim == 2:  # If grayscale, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Improve detection with histogram equalization
        equalized_gray = cv2.equalizeHist(gray)
        
        # Try with multiple detection parameters for better results
        faces = detector(equalized_gray, 1)  # Use 1 for upsampling
        
        if len(faces) == 0:
            # Try again with different parameters if no faces found
            faces = detector(equalized_gray, 2)  # More upsampling
            
        if len(faces) == 0:
            # Last attempt with the original image
            faces = detector(gray, 1)
            
        if len(faces) == 0:
            logger.warning("No faces detected in the image")
            return None if not return_image else None
            
        if largest_only and len(faces) > 1:
            face = max(
                faces,
                key=lambda rect: (rect.right() - rect.left())
                * (rect.bottom() - rect.top()),
            )
            faces = [face]

        results = []
        
        for face in faces:
            try:
                # Store original face coordinates
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                face_coords = (x1, y1, x2, y2)
                
                # Get landmarks from the face
                shape = predictor(gray, face)
                landmarks = face_utils.shape_to_np(shape)
                
                # Align face using landmarks
                aligned_face = align_face(img, landmarks, desired_face_width)
                
                if aligned_face is None:
                    logger.warning(f"Failed to align face")
                    continue

                # Re-detect landmarks on aligned face for better accuracy
                aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
                aligned_gray = cv2.equalizeHist(aligned_gray)
                
                # For aligned face, use a rectangle covering the full face
                aligned_rect = dlib.rectangle(0, 0, desired_face_width-1, desired_face_width-1)
                aligned_shape = predictor(aligned_gray, aligned_rect)
                aligned_landmarks = face_utils.shape_to_np(aligned_shape)
                
                if return_image:
                    results.append((aligned_landmarks.flatten(), aligned_face, face_coords))
                else:
                    results.append(aligned_landmarks.flatten())
            except Exception as e:
                logger.error(f"Error processing face: {e}")
                continue
                
        return results if len(results) > 0 else None
    except Exception as e:
        logger.error(f"Error extracting landmarks: {e}")
        return None if not return_image else None


def flip_landmarks(landmarks, face_width=DESIRED_FACE_WIDTH):
    """
    Flips the landmarks horizontally using the FLIP_MAP.
    Expects a flattened array of 136 values.
    """
    try:
        pts = landmarks.reshape(-1, 2)
        flipped = np.copy(pts)
        # Flip the x-coordinate
        flipped[:, 0] = face_width - pts[:, 0]
        # Reorder points using the mapping
        reordered = np.zeros_like(flipped)
        for i in range(68):
            reordered[i] = flipped[FLIP_MAP[i]]
        return reordered.flatten()
    except Exception as e:
        logger.error(f"Error in flipping landmarks: {e}")
        return landmarks


def advanced_augment_image(image, landmarks, max_rotation=10,
                           max_shift=10, max_scale=0.1):
    """
    Applies a series of augmentations:
      - Horizontal flip (with proper landmark reordering)
      - Affine transformation (rotation, scale, and shift)
      - Brightness/contrast change
      - Additive Gaussian noise
    Returns the augmented image and updated landmarks.
    """
    try:
        aug_img = image.copy()
        aug_landmarks = landmarks.copy()

        # Horizontal flip with probability 0.5
        if random.random() < 0.5:
            aug_img = cv2.flip(aug_img, 1)
            aug_landmarks = flip_landmarks(
                aug_landmarks, face_width=aug_img.shape[1]
            )

        # Affine transformation: rotation, scaling, shifting
        h, w = aug_img.shape[:2]
        center = (w // 2, h // 2)
        angle = random.uniform(-max_rotation, max_rotation)
        scale = random.uniform(1.0 - max_scale, 1.0 + max_scale)
        M_affine = cv2.getRotationMatrix2D(center, angle, scale)
        # Apply random shift
        tx = random.uniform(-max_shift, max_shift)
        ty = random.uniform(-max_shift, max_shift)
        M_affine[0, 2] += tx
        M_affine[1, 2] += ty
        aug_img = cv2.warpAffine(aug_img, M_affine, (w, h), flags=cv2.INTER_CUBIC)
        # Update landmarks with the affine transformation
        pts = aug_landmarks.reshape(-1, 2)
        pts_aug = []
        for (x, y) in pts:
            x_new = M_affine[0, 0] * x + M_affine[0, 1] * y + M_affine[0, 2]
            y_new = M_affine[1, 0] * x + M_affine[1, 1] * y + M_affine[1, 2]
            pts_aug.append([x_new, y_new])
        aug_landmarks = np.array(pts_aug).flatten()

        # Brightness and contrast adjustment
        alpha = random.uniform(0.8, 1.2)  # Contrast control
        beta = random.uniform(-20, 20)    # Brightness control
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

        # Add Gaussian noise
        noise_sigma = random.uniform(0, 10)
        noise = np.random.normal(0, noise_sigma, aug_img.shape).astype(np.float32)
        aug_img = cv2.add(aug_img.astype(np.float32), noise)
        aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)

        return aug_img, aug_landmarks
    except Exception as e:
        logger.error(f"Error in advanced augmentation: {e}")
        return image, landmarks


def create_training_dataset(
    data_dir, predictor_path, augmentation=True, augmentation_factor=2
):
    """
    Creates a training dataset from images stored in race-labeled directories.
    Applies landmark extraction and advanced augmentations as needed.
    Returns feature list X (flattened landmarks) and labels y.
    """
    X = []
    y = []
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()

    race_dirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    logger.info(f"Found {len(race_dirs)} race categories: {race_dirs}")

    for race_label in race_dirs:
        race_dir = os.path.join(data_dir, race_label)
        image_files = [
            f for f in os.listdir(race_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        logger.info(
            f"Processing {len(image_files)} images for race: {race_label}"
        )
        for image_file in tqdm(image_files, desc=f"Processing {race_label}"):
            image_path = os.path.join(race_dir, image_file)
            result = extract_facial_landmarks(
                image_path, predictor, detector,
                return_image=True, largest_only=True
            )
            if result is None:
                logger.warning(f"No face detected in {image_path}")
                continue
            landmarks, aligned_face, _ = result[0]  # Ignore face_coords during training
            if landmarks is None or aligned_face is None:
                logger.warning(f"Failed to process {image_path}")
                continue
            X.append(landmarks)
            y.append(race_label)
            # Data augmentation
            if augmentation:
                for _ in range(augmentation_factor):
                    aug_face, aug_landmarks = advanced_augment_image(
                        aligned_face, landmarks
                    )
                    X.append(aug_landmarks)
                    y.append(race_label)
    logger.info(f"Created dataset with {len(X)} samples")
    return X, y


def visualize_results(model, X_test, y_test, class_names):
    """
    Visualizes the confusion matrix and prints/saves the classification report.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")
    with open("classification_report.txt", "w") as f:
        f.write(report)


def train_model(X, y):
    """
    Trains the model using a pipeline that standardizes features,
    reduces dimensionality via PCA, and classifies with SVM.
    GridSearchCV is used for hyperparameter tuning.
    """
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
    )

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC(probability=True))
    ])
    param_grid = {
        # Use floats so PCA retains a percentage of the variance,
        # which avoids the issue of n_components > min(n_samples, n_features)
        "pca__n_components": [0.80, 0.85, 0.90, 0.95],
        "svm__C": [0.1, 1, 10, 100],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto", 0.1, 0.01],
    }

    logger.info("Starting hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    visualize_results(best_model, X_test, y_test, np.unique(y))
    return best_model, X_test, y_test


def visualize_predictions(image_path, predictions):
    """
    Visualizes predictions on the original image with bounding boxes, labels,
    and per-face probability bar charts with enhanced visual styling.
    """
    try:
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path.copy()
            
        # Modern color palette
        COLORS = {
            'background': '#EFF3F6',
            'title': '#2C3E50',
            'bbox': '#3498DB',
            'selected_bar': '#E74C3C',
            'bar': '#2ECC71',
            'text': '#FFFFFF',
            'face_border': '#F39C12',
            'grid': '#BDC3C7'
        }
        
        # Set the style for the plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        n_faces = len(predictions)
        
        # Create a figure with a nice background color
        fig = plt.figure(figsize=(15, max(5, 4 * (n_faces + 1))), facecolor=COLORS['background'])
        
        # Create a grid layout based on the number of faces
        gs = fig.add_gridspec(n_faces + 1, 2, height_ratios=[2] + [1] * n_faces)
        
        # Main image spans the top row
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(img)
        ax_main.set_title("Face Detection Results", 
                        fontsize=18, 
                        color=COLORS['title'],
                        fontweight='bold',
                        pad=10)
        ax_main.axis("off")
        
        # Add a subtle border to the main image
        for spine in ax_main.spines.values():
            spine.set_edgecolor(COLORS['grid'])
            spine.set_linewidth(2)
        
        # Display bounding boxes with more attractive styling
        for pred in predictions:
            x1, y1, x2, y2 = pred["face_coords"]
            race = pred["predicted_race"]
            conf = pred["confidence"]
            
            # Add a semi-transparent highlight area
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor=COLORS['bbox'],
                facecolor='none',
                linestyle='-',
                alpha=0.8
            )
            ax_main.add_patch(rect)
            
            # Add a visually appealing label
            label_bg = dict(
                boxstyle="round,pad=0.5",
                fc=COLORS['bbox'],
                ec=COLORS['bbox'],
                alpha=0.8
            )
            
            ax_main.text(
                x1, y1 - 10,
                f"Face {pred['face_id']}: {race.capitalize()} ({conf:.2f})",
                color=COLORS['text'],
                fontsize=12,
                fontweight='bold',
                bbox=label_bg,
                ha='left',
                va='bottom'
            )
        
        # Show each aligned face and its probability distribution with improved styling
        for i, pred in enumerate(predictions):
            # Left side: Aligned face with nice border
            ax_face = fig.add_subplot(gs[i+1, 0])
            ax_face.imshow(pred["aligned_face"])
            
            # Add decorative border around face
            for spine in ax_face.spines.values():
                spine.set_edgecolor(COLORS['face_border'])
                spine.set_linewidth(2)
                
            ax_face.set_title(
                f"Face {pred['face_id']} (Aligned)",
                fontsize=14,
                color=COLORS['title']
            )
            ax_face.axis("off")
            
            # Right side: Probability bars with better styling
            ax_prob = fig.add_subplot(gs[i+1, 1])
            
            # Sort probabilities for better visualization
            races = [p[0] for p in pred["all_probabilities"]]
            probs = [p[1] for p in pred["all_probabilities"]]
            
            # Create horizontal bars
            bars = ax_prob.barh(
                races,
                probs,
                color=COLORS['bar'],
                height=0.6,
                alpha=0.8,
                edgecolor='none'
            )
            
            # Add percentage labels to the bars
            for j, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width + 0.01
                ax_prob.text(
                    label_x_pos,
                    bar.get_y() + bar.get_height()/2,
                    f"{probs[j]:.1%}",
                    va='center',
                    fontweight='bold',
                    color=COLORS['title']
                )
            
            # Highlight the predicted race
            idx = races.index(pred["predicted_race"])
            bars[idx].set_color(COLORS['selected_bar'])
            
            # Add a label to indicate which one was selected
            ax_prob.text(
                probs[idx] / 2,
                idx,
                "PREDICTED",
                ha='center',
                va='center',
                color='white',
                fontweight='bold',
                fontsize=10
            )
            
            # Better formatting
            ax_prob.set_xlim(0, 1.15)  # Add space for percentage labels
            ax_prob.set_title(
                f"Ethnic Group Probability Distribution",
                fontsize=14,
                color=COLORS['title']
            )
            ax_prob.set_xlabel("Probability", fontsize=12, color=COLORS['title'])
            
            # Make race labels nicely capitalized
            ax_prob.set_yticks(range(len(races)))
            ax_prob.set_yticklabels([r.capitalize() for r in races])
            
            # Style the grid for better visibility
            ax_prob.grid(True, axis='x', linestyle='--', alpha=0.7)
            ax_prob.set_axisbelow(True)
            
            # Remove top and right spines for cleaner look
            ax_prob.spines['top'].set_visible(False)
            ax_prob.spines['right'].set_visible(False)
        
        # Add a title and timestamp
        plt.suptitle(
            "Ethnicity Detection Results", 
            fontsize=22, 
            y=0.98, 
            color=COLORS['title'],
            fontweight='bold'
        )
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.5, 0.01,
            f"Analysis completed: {timestamp}",
            ha='center',
            fontsize=10,
            color=COLORS['title'],
            fontstyle='italic'
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.862, bottom=0.05)  # Make room for title and timestamp
        
        # Save high-quality visualization
        plt.savefig(
            "prediction_visualization.png",
            dpi=300,
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )
        
        plt.show()
    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}")
        logger.debug(f"Detailed error information:", exc_info=True)


def predict_race(image_path, predictor_path, model_path, visualize=False):
    """
    Predicts the race of faces in the given image using the trained model.
    Returns a list of predictions.
    """
    try:
        model = joblib.load(model_path)
        predictor = dlib.shape_predictor(predictor_path)
        detector = dlib.get_frontal_face_detector()
        
        # Load the image
        if isinstance(image_path, str):
            img_original = cv2.imread(image_path)
            if img_original is None:
                logger.error(f"Failed to read image: {image_path}")
                return []
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        else:
            img_original = image_path.copy()
        
        # Extract landmarks with improved face detection
        results = extract_facial_landmarks(
            img_original, predictor, detector, return_image=True, largest_only=False
        )
        
        if not results:
            logger.warning("No faces detected or processed in the image.")
            return []
            
        predictions = []
        for idx, result in enumerate(results):
            landmarks, aligned_face, face_coords = result
            if landmarks is None or aligned_face is None:
                continue
                
            # Check if landmarks contain NaN values
            if np.isnan(landmarks).any():
                logger.warning(f"Face {idx} has invalid landmark values (NaN)")
                continue
                
            landmarks_reshaped = landmarks.reshape(1, -1)
            predicted_race = model.predict(landmarks_reshaped)[0]
            probabilities = model.predict_proba(landmarks_reshaped)[0]
            
            # Create map of race to probability
            race_probs = {
                race: prob
                for race, prob in zip(model.classes_, probabilities)
            }
            
            # Sort probabilities from highest to lowest
            sorted_probs = sorted(
                race_probs.items(), key=lambda x: x[1], reverse=True
            )
            
            prediction = {
                "face_id": idx,
                "predicted_race": predicted_race,
                "confidence": race_probs[predicted_race],
                "all_probabilities": sorted_probs,
                "face_coords": face_coords,  # Using actual face coordinates
                "aligned_face": aligned_face,
            }
            predictions.append(prediction)
            
        if visualize and predictions:
            visualize_predictions(img_original, predictions)
            
        return predictions
    except Exception as e:
        logger.error(f"Error predicting race: {e}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced Race Detection Model: Train or Predict"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        required=True,
        help="Mode: train or predict",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory for training data (if mode=train)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to image for prediction (if mode=predict)",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to facial landmark predictor",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="race_detection_model.pkl",
        help="Path to model file for prediction or to save the model during training",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize predictions"
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable data augmentation (default: enabled)",
    )
    parser.add_argument(
        "--aug_factor",
        type=int,
        default=2,
        help="Data augmentation factor (default: 2)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        if not args.data_dir:
            logger.error("Training mode requires --data_dir")
            exit(1)
        if not os.path.exists(args.predictor):
            logger.error(
                f"Predictor file not found: {args.predictor}. Please download it from dlib."
            )
            exit(1)
        if not os.path.exists(args.data_dir):
            logger.error(f"Data directory not found: {args.data_dir}")
            exit(1)
        logger.info("Starting dataset creation...")
        X, y = create_training_dataset(
            args.data_dir,
            args.predictor,
            augmentation=args.augmentation,
            augmentation_factor=args.aug_factor,
        )
        if len(X) == 0:
            logger.error("No training data found or processed.")
            exit(1)
        logger.info("Starting model training...")
        model, X_test, y_test = train_model(X, y)
        joblib.dump(model, args.model)
        logger.info(f"Trained model saved to {args.model}")
        metadata = {
            "classes": list(np.unique(y)),
            "feature_count": X[0].shape[0],
            "training_samples": len(X),
            "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
        }
        joblib.dump(metadata, "model_metadata.pkl")
        logger.info("Model training complete!")
    elif args.mode == "predict":
        if not args.image_path:
            logger.error("Prediction mode requires --image_path")
            exit(1)
        if not os.path.exists(args.image_path):
            logger.error(f"Image file not found: {args.image_path}")
            exit(1)
        if not os.path.exists(args.predictor):
            logger.error(
                f"Predictor file not found: {args.predictor}. Please download it from dlib."
            )
            exit(1)
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            exit(1)
        predictions = predict_race(
            args.image_path, args.predictor, args.model, visualize=args.visualize
        )
        if not predictions:
            print("No faces detected or processed in the image.")
        else:
            print(f"Found {len(predictions)} faces in the image:")
            for pred in predictions:
                print(f"\nFace {pred['face_id']}:")
                print(f"  Predicted race: {pred['predicted_race']}")
                print(f"  Confidence: {pred['confidence']:.4f}")
                print("  All probabilities:")
                for race, prob in pred["all_probabilities"]:
                    print(f"    {race}: {prob:.4f}")
    else:
        logger.error("Invalid mode selected.")
