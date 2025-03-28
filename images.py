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
import time  # Import time for FPS calculation
import datetime # For visualization timestamp

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
# fmt: off
FLIP_MAP = {
    0: 16, 1: 15, 2: 14, 3: 13, 4: 12, 5: 11, 6: 10, 7: 9, 8: 8,
    9: 7, 10: 6, 11: 5, 12: 4, 13: 3, 14: 2, 15: 1, 16: 0,
    17: 26, 18: 25, 19: 24, 20: 23, 21: 22, 22: 21, 23: 20, 24: 19, 25: 18, 26: 17,
    27: 27, 28: 28, 29: 29, 30: 30, # Nose bridge points
    31: 35, 32: 34, 33: 33, 34: 32, 35: 31, # Lower nose points
    36: 45, 37: 44, 38: 43, 39: 42, 40: 47, 41: 46, # Left eye -> Right eye
    42: 39, 43: 38, 44: 37, 45: 36, 46: 41, 47: 40, # Right eye -> Left eye
    48: 54, 49: 53, 50: 52, 51: 51, 52: 50, 53: 49, 54: 48, # Outer mouth
    55: 59, 56: 58, 57: 57, 58: 56, 59: 55, # Inner mouth
    60: 64, 61: 63, 62: 62, 63: 61, 64: 60, # Inner mouth bottom
    65: 67, 66: 66, 67: 65 # Inner mouth corners
}
# fmt: on


def align_face(image, landmarks, desired_face_width=256, desired_eye_dist_factor=0.35):
    """
    Aligns and crops a face based on facial landmarks.
    Includes improved error handling and validation.
    Expects image in RGB format.
    """
    try:
        # Validate inputs
        if image is None:
            logger.error("Invalid input: image is None")
            return None
        if landmarks is None:
             logger.error("Invalid input: landmarks is None")
             return None

        # Create a copy of the image to avoid modification of the original
        image_rgb = image.copy()
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            logger.error(f"Invalid image format for alignment: expected 3 channels (RGB), got shape {image_rgb.shape}")
            # Attempt conversion if possible (e.g., grayscale)
            if len(image_rgb.shape) == 2:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            else:
                return None # Cannot handle other formats reliably here

        # Convert landmarks to float32 for precise calculations
        landmarks = landmarks.astype(np.float32)

        # Check for NaN values in landmarks
        if np.isnan(landmarks).any():
            logger.error("Invalid landmarks: contains NaN values")
            return None

        # Calculate the average positions for the left and right eyes (dlib indices)
        left_eye_pts = landmarks[36:42]
        right_eye_pts = landmarks[42:48]

        # Validate eye landmarks
        if left_eye_pts.shape[0] == 0 or right_eye_pts.shape[0] == 0:
            logger.error("Invalid eye landmarks (empty arrays)")
            return None

        # Calculate eye centers with validation
        left_eye_center = np.mean(left_eye_pts, axis=0)
        right_eye_center = np.mean(right_eye_pts, axis=0)

        # Ensure eye coordinates are valid
        if np.isnan(left_eye_center).any() or np.isnan(right_eye_center).any():
            logger.error("Invalid eye coordinates: NaN values detected after mean calculation")
            return None

        # Calculate the angle between the eyes
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        dist = np.sqrt((dX**2) + (dY**2))

        # Avoid division by zero or near-zero distance
        if dist < 1e-6:
            logger.warning("Eye distance too small or zero, cannot calculate angle reliably.")
            # Default to no rotation if eyes are too close
            angle = 0.0
        else:
            angle = np.degrees(np.arctan2(dY, dX))

        # Calculate the desired distance between eyes based on the desired face width
        desired_dist = desired_eye_dist_factor * desired_face_width
        # Calculate the scale needed to achieve the desired distance
        # Handle case where original distance is very small
        if dist < 1e-6:
             scale = 1.0 # Avoid division by zero, default scale
             logger.warning("Eye distance near zero, using default scale 1.0")
        else:
             scale = desired_dist / dist

        # Validate scale factor (avoid excessively large or small scales)
        if not (0.1 < scale < 10.0): # Adjust bounds as needed
            logger.warning(f"Unusual scale factor calculated: {scale:.4f}. Clamping or review data.")
            # Optional: Clamp scale or return None depending on desired strictness
            # scale = np.clip(scale, 0.1, 10.0)
            # return None # More strict

        # Calculate the center point between the eyes
        eyes_center = (
            (left_eye_center[0] + right_eye_center[0]) / 2.0,
            (left_eye_center[1] + right_eye_center[1]) / 2.0,
        )

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update the translation component of the matrix to center the face
        # Adjust tX, tY to position the center of the eyes appropriately in the output image
        tX = desired_face_width * 0.5
        tY = desired_face_width * desired_eye_dist_factor # Position eyes vertically based on factor
        M[0, 2] += tX - eyes_center[0]
        M[1, 2] += tY - eyes_center[1]

        # Apply the affine transformation
        output_size = (desired_face_width, desired_face_width)
        aligned_face = cv2.warpAffine(
            image_rgb, # Use the RGB image
            M,
            output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0), # Black border
        )

        # Validate output
        if aligned_face is None or aligned_face.size == 0:
            logger.error("Failed to create aligned face after warpAffine")
            return None

        return aligned_face

    except Exception as e:
        logger.error(f"Error in face alignment: {str(e)}", exc_info=True)
        return None


def extract_facial_landmarks(
    image_input,  # Can be path or numpy array
    predictor,
    detector,
    desired_face_width=DESIRED_FACE_WIDTH,
    return_image=False,
    largest_only=False,
):
    """
    Extracts facial landmarks from an image (path or array).
    Improves detection, handles color spaces, aligns face, and re-detects
    landmarks on the aligned face for consistency.
    Returns list of tuples: (aligned_landmarks_flat, aligned_face_rgb, original_face_coords)
    or None if no faces are processed.
    """
    try:
        img_rgb = None
        if isinstance(image_input, str):
            # Use skimage.io for broader format support, convert to RGB
            img_read = io.imread(image_input)
            if img_read is None:
                logger.error(f"Failed to read image: {image_input}")
                return None
            if img_read.ndim == 2: # Grayscale
                img_rgb = cv2.cvtColor(img_read, cv2.COLOR_GRAY2RGB)
            elif img_read.ndim == 3:
                if img_read.shape[2] == 4: # RGBA
                    img_rgb = cv2.cvtColor(img_read, cv2.COLOR_RGBA2RGB)
                elif img_read.shape[2] == 3: # Assume RGB
                    img_rgb = img_read
                else:
                    logger.error(f"Unsupported number of channels: {img_read.shape[2]} in {image_input}")
                    return None
            else:
                 logger.error(f"Unsupported image dimensions: {img_read.ndim} in {image_input}")
                 return None

        elif isinstance(image_input, np.ndarray):
            img_input_copy = image_input.copy() # Work with a copy
            if img_input_copy.ndim == 2: # Grayscale
                img_rgb = cv2.cvtColor(img_input_copy, cv2.COLOR_GRAY2RGB)
            elif img_input_copy.ndim == 3:
                if img_input_copy.shape[2] == 3: # Assume BGR from OpenCV camera feed
                    img_rgb = cv2.cvtColor(img_input_copy, cv2.COLOR_BGR2RGB)
                elif img_input_copy.shape[2] == 4: # RGBA
                    img_rgb = cv2.cvtColor(img_input_copy, cv2.COLOR_RGBA2RGB)
                else:
                    logger.error(f"Unsupported number of channels in numpy array: {img_input_copy.shape[2]}")
                    return None
            else:
                 logger.error(f"Unsupported numpy array dimensions: {img_input_copy.ndim}")
                 return None
        else:
            logger.error(f"Invalid image input type: {type(image_input)}")
            return None

        if img_rgb is None: # Should not happen if logic above is correct, but check
             logger.error("Image conversion to RGB failed.")
             return None

        # Convert to grayscale for detection
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Improve detection with histogram equalization
        equalized_gray = cv2.equalizeHist(gray)

        # Try detection with upsampling for potentially better results
        faces = detector(equalized_gray, 1)  # Use 1 for upsampling

        # If no faces found, try again with more upsampling or original gray
        if len(faces) == 0:
            # logger.debug("No faces found with equalized_gray + upsample=1. Trying upsample=2.")
            faces = detector(equalized_gray, 2) # More upsampling
        if len(faces) == 0:
            # logger.debug("No faces found with equalized_gray + upsample=2. Trying original gray.")
            faces = detector(gray, 1) # Try original gray image

        if len(faces) == 0:
            # logger.warning("No faces detected in the image after multiple attempts.") # Can be noisy
            return None

        if largest_only and len(faces) > 1:
            face = max(
                faces,
                key=lambda rect: (rect.right() - rect.left())
                * (rect.bottom() - rect.top()),
            )
            faces = [face]

        results = []
        img_h, img_w = img_rgb.shape[:2]

        for face_rect in faces:
            try:
                # Store original face coordinates (ensure they are within image bounds)
                x1 = max(0, face_rect.left())
                y1 = max(0, face_rect.top())
                x2 = min(img_w, face_rect.right())
                y2 = min(img_h, face_rect.bottom())
                original_face_coords = (x1, y1, x2, y2)

                # Check if face rectangle is valid after clamping
                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"Invalid face rectangle detected or clamped to zero size: {original_face_coords}")
                    continue

                # --- Step 1: Get landmarks from the original detected face region ---
                # Use the original grayscale image for this initial landmark detection
                shape_orig = predictor(gray, face_rect)
                landmarks_orig = face_utils.shape_to_np(shape_orig)

                # --- Step 2: Align the face using the original landmarks ---
                # Pass the full color (RGB) image and original landmarks
                aligned_face_rgb = align_face(img_rgb, landmarks_orig, desired_face_width)

                if aligned_face_rgb is None:
                    logger.warning(f"Failed to align face for rect {original_face_coords}")
                    continue

                # --- Step 3: Re-detect landmarks on the ALIGNED face ---
                # This is crucial for consistency with training data
                aligned_gray = cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2GRAY)
                # Optional: Equalize the aligned face too
                aligned_gray_eq = cv2.equalizeHist(aligned_gray)

                # Define the rectangle for the *entire* aligned face
                aligned_rect = dlib.rectangle(
                    0, 0, desired_face_width - 1, desired_face_width - 1
                )
                # Detect landmarks on the equalized aligned grayscale face
                shape_aligned = predictor(aligned_gray_eq, aligned_rect)
                aligned_landmarks = face_utils.shape_to_np(shape_aligned)

                # Ensure aligned landmarks are valid before adding
                if np.isnan(aligned_landmarks).any():
                    logger.warning("NaN values found in re-detected aligned landmarks, skipping face.")
                    continue

                # Flatten the ALIGNED landmarks for the model
                aligned_landmarks_flat = aligned_landmarks.flatten()

                # Store results
                if return_image:
                    results.append(
                        (
                            aligned_landmarks_flat,
                            aligned_face_rgb, # Return the aligned RGB face
                            original_face_coords, # Return original bounding box
                        )
                    )
                else:
                    # If only features are needed (e.g., during bulk training data creation)
                    results.append(aligned_landmarks_flat)

            except Exception as e:
                logger.error(f"Error processing individual face at {face_rect}: {e}", exc_info=True)
                continue # Skip this face

        return results if results else None # Return None if list is empty

    except Exception as e:
        logger.error(f"Error in extract_facial_landmarks: {e}", exc_info=True)
        return None


def flip_landmarks(landmarks_flat, face_width=DESIRED_FACE_WIDTH):
    """
    Flips the landmarks horizontally using the FLIP_MAP.
    Expects a flattened array of 136 values (68 points * 2 coords).
    """
    if landmarks_flat is None or landmarks_flat.shape[0] != 136:
        logger.error(f"Invalid input for flip_landmarks. Expected flattened 136 values, got shape {landmarks_flat.shape if landmarks_flat is not None else 'None'}")
        return landmarks_flat # Return original if invalid

    try:
        pts = landmarks_flat.reshape(68, 2)
        flipped_pts = np.copy(pts)

        # Flip the x-coordinate
        flipped_pts[:, 0] = face_width - 1 - pts[:, 0] # Use width-1 as max index

        # Reorder points using the mapping
        reordered_pts = np.zeros_like(flipped_pts)
        for i in range(68):
            map_index = FLIP_MAP.get(i)
            if map_index is None:
                logger.warning(f"Index {i} not found in FLIP_MAP. Landmark will not be reordered correctly.")
                reordered_pts[i] = flipped_pts[i] # Keep original if map fails
            else:
                reordered_pts[i] = flipped_pts[map_index]

        return reordered_pts.flatten()
    except Exception as e:
        logger.error(f"Error in flipping landmarks: {e}", exc_info=True)
        return landmarks_flat # Return original on error


def advanced_augment_image(
    image_rgb, landmarks_flat, max_rotation=10, max_shift_px=10, scale_range=(0.9, 1.1),
    brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), noise_sigma_max=10
):
    """
    Applies a series of augmentations to an RGB image and its flattened landmarks:
      - Horizontal flip (with proper landmark reordering)
      - Affine transformation (rotation, scale, and shift)
      - Brightness/contrast change
      - Additive Gaussian noise
    Returns the augmented RGB image and updated flattened landmarks.
    Assumes input landmarks correspond to the input image dimensions.
    """
    try:
        aug_img = image_rgb.copy()
        aug_landmarks_flat = landmarks_flat.copy()
        h, w = aug_img.shape[:2]

        # 1. Horizontal flip with probability 0.5
        if random.random() < 0.5:
            aug_img = cv2.flip(aug_img, 1) # Flip horizontally
            aug_landmarks_flat = flip_landmarks(aug_landmarks_flat, face_width=w)

        # 2. Affine transformation: rotation, scaling, shifting
        center = (w // 2, h // 2)
        angle = random.uniform(-max_rotation, max_rotation)
        scale = random.uniform(scale_range[0], scale_range[1])
        M_affine = cv2.getRotationMatrix2D(center, angle, scale)

        # Apply random shift (translation)
        tx = random.uniform(-max_shift_px, max_shift_px)
        ty = random.uniform(-max_shift_px, max_shift_px)
        M_affine[0, 2] += tx
        M_affine[1, 2] += ty

        # Warp the image
        aug_img = cv2.warpAffine(
            aug_img, M_affine, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        # Update landmarks with the affine transformation
        pts = aug_landmarks_flat.reshape(68, 2)
        # Add column of 1s for matrix multiplication
        pts_homogeneous = np.hstack((pts, np.ones((68, 1))))
        # Apply the affine transformation matrix (first 2 rows)
        pts_transformed = M_affine @ pts_homogeneous.T
        aug_landmarks_flat = pts_transformed.T.flatten()

        # 3. Brightness and contrast adjustment
        alpha = random.uniform(contrast_range[0], contrast_range[1]) # Contrast
        beta = random.uniform(-20, 20) # Brightness (absolute value)
        # Alternative brightness relative to range:
        # beta = random.uniform(brightness_range[0] * 127, brightness_range[1] * 127) - 127
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

        # 4. Add Gaussian noise
        noise_sigma = random.uniform(0, noise_sigma_max)
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, aug_img.shape).astype(np.float32)
            # Add noise and clip to valid range
            aug_img_float = aug_img.astype(np.float32) + noise
            aug_img = np.clip(aug_img_float, 0, 255).astype(np.uint8)

        # Ensure landmarks are still valid (e.g., not NaN after transforms)
        if np.isnan(aug_landmarks_flat).any():
             logger.warning("NaN values generated in augmented landmarks. Returning original.")
             return image_rgb, landmarks_flat # Revert if augmentation broke landmarks

        return aug_img, aug_landmarks_flat

    except Exception as e:
        logger.error(f"Error in advanced augmentation: {e}", exc_info=True)
        return image_rgb, landmarks_flat # Return original on error


def create_training_dataset(
    data_dir, predictor_path, augmentation=True, augmentation_factor=2
):
    """
    Creates a training dataset from images stored in race-labeled directories.
    Applies landmark extraction (on aligned faces) and advanced augmentations.
    Returns feature list X (flattened landmarks) and labels y.
    """
    X = []
    y = []
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()

    race_dirs = [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    if not race_dirs:
        logger.error(f"No subdirectories found in data directory: {data_dir}")
        return [], []
    logger.info(f"Found {len(race_dirs)} race categories: {race_dirs}")

    total_images = 0
    processed_images = 0
    augmented_samples = 0

    for race_label in race_dirs:
        race_dir = os.path.join(data_dir, race_label)
        try:
            image_files = [
                f
                for f in os.listdir(race_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]
        except OSError as e:
            logger.error(f"Could not read directory {race_dir}: {e}")
            continue

        logger.info(
            f"Processing {len(image_files)} images for race: {race_label}"
        )
        total_images += len(image_files)

        for image_file in tqdm(image_files, desc=f"Processing {race_label}", unit="img"):
            image_path = os.path.join(race_dir, image_file)
            try:
                # Pass image path to extract_facial_landmarks
                # Request aligned face image for augmentation
                result = extract_facial_landmarks(
                    image_path,
                    predictor,
                    detector,
                    return_image=True,
                    largest_only=True, # Use only largest face per image for training
                )

                if result is None:
                    # logger.warning(f"No face detected or processed in {image_path}") # Too verbose
                    continue

                # result is a list, take the first element (largest_only=True)
                # landmarks_flat are from the *aligned* face
                landmarks_flat, aligned_face_rgb, _ = result[0]

                if landmarks_flat is None or aligned_face_rgb is None:
                    logger.warning(f"Failed to get valid landmarks/aligned face from {image_path}")
                    continue

                # Check for NaN before adding original
                if np.isnan(landmarks_flat).any():
                    logger.warning(f"Original landmarks for {image_path} contain NaN. Skipping.")
                    continue

                X.append(landmarks_flat)
                y.append(race_label)
                processed_images += 1

                # Data augmentation
                if augmentation:
                    for _ in range(augmentation_factor):
                        try:
                            # Augment the ALIGNED face and its corresponding landmarks
                            aug_face, aug_landmarks_flat = advanced_augment_image(
                                aligned_face_rgb, landmarks_flat
                            )

                            # Ensure augmented landmarks are valid before adding
                            if not np.isnan(aug_landmarks_flat).any():
                                X.append(aug_landmarks_flat)
                                y.append(race_label)
                                augmented_samples += 1
                            else:
                                logger.warning(f"NaN values in augmented landmarks for {image_path}, skipping augmentation instance.")
                        except Exception as aug_err:
                             logger.error(f"Error during augmentation for {image_path}: {aug_err}", exc_info=True)

            except Exception as proc_err:
                logger.error(f"Error processing image file {image_path}: {proc_err}", exc_info=True)

    logger.info(f"Finished dataset creation. Processed {processed_images}/{total_images} images.")
    logger.info(f"Generated {len(X)} total samples ({augmented_samples} augmented).")
    if len(X) == 0:
        logger.error("No valid samples were generated. Check data directory and image files.")

    return X, y


def visualize_results(model, X_test, y_test, class_names, output_dir="."):
    """
    Visualizes the confusion matrix and prints/saves the classification report.
    Saves outputs to the specified directory.
    """
    logger.info("Generating test set predictions for visualization...")
    try:
        y_pred = model.predict(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=class_names
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
        ax.set_title("Confusion Matrix")
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {cm_path}")
        plt.close(fig) # Close the plot

        # Classification Report
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        logger.info(f"Classification Report:\n{report}")
        report_path = os.path.join(output_dir, "classification_report.txt")
        try:
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Classification report saved to {report_path}")
        except IOError as e:
            logger.error(f"Failed to save classification report: {e}")

    except Exception as e:
        logger.error(f"Error during results visualization: {e}", exc_info=True)


def train_model(X, y, model_save_path="race_detection_model.pkl", metadata_save_path="model_metadata.pkl"):
    """
    Trains the model using a pipeline (Scaler, PCA, SVM).
    Uses GridSearchCV for hyperparameter tuning.
    Handles NaN values, saves the model and metadata.
    Returns the best model, X_test, y_test. Returns None if training fails.
    """
    if not X or not y:
        logger.error("Training data (X or y) is empty. Cannot train model.")
        return None, None, None

    try:
        X = np.array(X, dtype=np.float64) # Use float64 for stability
        y = np.array(y)
    except ValueError as e:
        logger.error(f"Could not convert training data to numpy arrays: {e}. Check data consistency.")
        return None, None, None

    # Check for NaN/inf values BEFORE splitting
    nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
    if np.any(nan_mask):
        num_nan = np.sum(nan_mask)
        logger.warning(f"Found {num_nan} samples with NaN/inf landmarks. Removing them.")
        X = X[~nan_mask]
        y = y[~nan_mask]
        if X.shape[0] == 0:
            logger.error("No valid training data left after removing NaN/inf samples.")
            return None, None, None

    if X.shape[0] < 10: # Arbitrary small number check
        logger.error(f"Very few training samples ({X.shape[0]}). Training may be unreliable.")
        # Decide whether to proceed or exit based on requirements
        # return None, None, None # Stricter check

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
         logger.error(f"Error during train/test split (likely too few samples per class for stratify): {e}")
         # Fallback to non-stratified split if needed, though less ideal
         logger.warning("Falling back to non-stratified train/test split.")
         X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
         )


    logger.info(
        f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
    )
    class_names = np.unique(y)
    logger.info(f"Classes: {class_names}")

    # Define the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        # PCA n_components can be int or float (variance explained)
        # Start with float for flexibility
        ("pca", PCA(random_state=42)),
        ("svm", SVC(probability=True, random_state=42, class_weight='balanced')), # Added balanced weights
    ])

    # Define the parameter grid for GridSearchCV
    # Adjust n_components based on expected feature dimensions (136)
    # If PCA variance is high, fewer components might suffice
    n_features = X_train.shape[1]
    pca_components_options = [0.85, 0.90, 0.95]
    # Add integer options if variance floats don't work well or for comparison
    # Ensure integers are <= n_features
    pca_components_options.extend([min(n, n_features) for n in [30, 50, 70] if n <= n_features])
    # Remove duplicates and sort
    pca_components_options = sorted(list(set(pca_components_options)))


    param_grid = {
        "pca__n_components": pca_components_options,
        "svm__C": [0.1, 1, 10, 50, 100], # Expanded C range
        "svm__kernel": ["linear", "rbf"],
        # Gamma only relevant for 'rbf' kernel, GridSearchCV handles this
        "svm__gamma": ["scale", "auto", 0.01, 0.001], # Expanded gamma range
    }

    logger.info("Starting hyperparameter tuning with GridSearchCV...")
    logger.info(f"Parameter grid: {param_grid}")
    try:
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1, error_score='raise'
        )
        grid_search.fit(X_train, y_train)
    except ValueError as e:
        logger.error(f"Error during GridSearchCV fitting: {e}. Check PCA components vs features or SVM parameters.", exc_info=True)
        return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during GridSearchCV: {e}", exc_info=True)
        return None, None, None


    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test set accuracy: {test_accuracy:.4f}")

    # Visualize results (Confusion Matrix, Classification Report)
    visualize_results(best_model, X_test, y_test, class_names)

    # Save the trained model
    try:
        joblib.dump(best_model, model_save_path)
        logger.info(f"Trained model saved to {model_save_path}")
    except Exception as e:
        logger.error(f"Failed to save trained model: {e}", exc_info=True)
        # Return the model anyway, but log the failure
        # return None, None, None # Stricter: fail if cannot save

    # Save metadata
    try:
        # Get actual number of PCA components used by the best model
        n_pca_components = best_model.named_steps['pca'].n_components_
        if isinstance(n_pca_components, float): # If it was variance ratio
             n_pca_components = best_model.named_steps['pca'].n_components # Get the resulting int

        metadata = {
            "classes": list(class_names),
            "input_feature_count": X_train.shape[1], # Features before PCA
            "pca_components_used": n_pca_components, # Features after PCA
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "cross_val_accuracy": grid_search.best_score_,
            "test_accuracy": test_accuracy,
            "best_params": grid_search.best_params_,
            "pipeline_steps": list(best_model.named_steps.keys()),
        }
        joblib.dump(metadata, metadata_save_path)
        logger.info(f"Model metadata saved to {metadata_save_path}")
    except Exception as e:
        logger.error(f"Failed to save model metadata: {e}", exc_info=True)

    return best_model, X_test, y_test


def visualize_predictions(image_or_path, predictions, output_path="prediction_visualization.png"):
    """
    Visualizes predictions on the original image with bounding boxes, labels,
    and per-face probability bar charts with enhanced visual styling.
    Handles both image path and numpy array (RGB expected) input.
    """
    try:
        img_display = None
        if isinstance(image_or_path, str):
            # Read with OpenCV, convert BGR to RGB for matplotlib
            img_bgr = cv2.imread(image_or_path)
            if img_bgr is None:
                logger.error(f"Failed to read image for visualization: {image_or_path}")
                return
            img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_or_path, np.ndarray):
            # Assume input is RGB (as processed internally) or BGR (from camera maybe)
            img_input_copy = image_or_path.copy()
            if len(img_input_copy.shape) == 3 and img_input_copy.shape[2] == 3:
                 # Heuristic: If it looks like BGR (common from cv2 reads), convert.
                 # This might need adjustment if the source is guaranteed RGB.
                 # Let's assume it might be BGR and convert. If it was already RGB,
                 # the colors might flip, but detection used RGB anyway.
                 # A safer approach might be to require RGB input here.
                 # For now, let's assume the input *might* be BGR.
                 # img_display = cv2.cvtColor(img_input_copy, cv2.COLOR_BGR2RGB)
                 # --- Correction: Assume the input `image_or_path` is the RGB
                 # --- version used in `predict_race` if it's an array.
                 img_display = img_input_copy
            elif len(img_input_copy.shape) == 2:
                 img_display = cv2.cvtColor(img_input_copy, cv2.COLOR_GRAY2RGB)
            else:
                 logger.error(f"Invalid numpy array shape for visualization: {img_input_copy.shape}")
                 return
        else:
            logger.error(f"Invalid image input type for visualization: {type(image_or_path)}")
            return

        if img_display is None:
            logger.error("Image for visualization is None after processing.")
            return

        # Modern color palette (adjust as desired)
        COLORS = {
            "background": "#F8F9FA", # Light gray background
            "title": "#212529",      # Dark text
            "bbox": "#007BFF",       # Blue bounding box
            "selected_bar": "#DC3545", # Red for predicted
            "bar": "#28A745",       # Green for other bars
            "text_light": "#FFFFFF", # White text on dark backgrounds
            "text_dark": "#343A40",  # Dark text for labels
            "face_border": "#FFC107", # Yellow border for aligned face
            "grid": "#CED4DA",       # Light grid lines
        }

        plt.style.use("seaborn-v0_8-whitegrid") # Use a clean style

        n_faces = len(predictions)
        if n_faces == 0:
            logger.info("No predictions to visualize.")
            # Optionally display the image anyway
            # fig, ax = plt.subplots(figsize=(10, 8))
            # ax.imshow(img_display)
            # ax.set_title("No Faces Detected", color=COLORS['title'])
            # ax.axis('off')
            # plt.show()
            return

        # Create figure and grid layout
        # Height: 2 units for main image, 1 unit per face plot row
        fig = plt.figure(
            figsize=(16, 5 + 3 * n_faces), # Adjusted size
            facecolor=COLORS["background"],
        )
        gs = fig.add_gridspec(
            n_faces + 1, 2, height_ratios=[2] + [1] * n_faces, wspace=0.3, hspace=0.5
        )

        # --- Main Image Display ---
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(img_display)
        ax_main.set_title(
            "Ethnicity Prediction Results",
            fontsize=18, color=COLORS["title"], fontweight="bold", pad=15,
        )
        ax_main.axis("off")

        # Draw bounding boxes and labels on main image
        for pred in predictions:
            x1, y1, x2, y2 = pred["face_coords"]
            race = pred["predicted_race"]
            conf = pred["confidence"]
            face_id = pred["face_id"]

            # Bounding box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=COLORS["bbox"], facecolor="none", alpha=0.9
            )
            ax_main.add_patch(rect)

            # Label background
            label_bg = dict(boxstyle="round,pad=0.3", fc=COLORS["bbox"], ec="none", alpha=0.8)
            # Label text
            ax_main.text(
                x1 + 5, y1 - 8, # Position slightly inside top-left
                f"Face {face_id}: {race.capitalize()} ({conf:.2f})",
                color=COLORS["text_light"], fontsize=10, fontweight="bold",
                bbox=label_bg, ha="left", va="bottom",
            )

        # --- Per-Face Details ---
        for i, pred in enumerate(predictions):
            face_id = pred["face_id"]
            aligned_face_rgb = pred["aligned_face"] # Should be RGB
            all_probs = pred["all_probabilities"] # List of (race, prob) tuples
            predicted_race = pred["predicted_race"]

            # --- Aligned Face (Left Column) ---
            ax_face = fig.add_subplot(gs[i + 1, 0])
            if aligned_face_rgb is not None:
                 # Ensure it's displayable (correct type and channels)
                 if aligned_face_rgb.ndim == 3 and aligned_face_rgb.shape[2] == 3:
                     ax_face.imshow(aligned_face_rgb)
                 else:
                     ax_face.text(0.5, 0.5, 'Invalid\nFormat', ha='center', va='center')
                     logger.warning(f"Aligned face for Face {face_id} has unexpected shape: {aligned_face_rgb.shape}")
            else:
                 ax_face.text(0.5, 0.5, 'Not Available', ha='center', va='center')

            ax_face.set_title(f"Face {face_id} (Aligned)", fontsize=12, color=COLORS["title"])
            ax_face.axis("off")
            # Add border
            for spine in ax_face.spines.values():
                spine.set_edgecolor(COLORS["face_border"])
                spine.set_linewidth(2.5)
                spine.set_visible(True)

            # --- Probability Distribution (Right Column) ---
            ax_prob = fig.add_subplot(gs[i + 1, 1])

            if not all_probs:
                ax_prob.text(0.5, 0.5, 'Probabilities\nNot Available', ha='center', va='center')
                ax_prob.set_title(f"Face {face_id} Probabilities", fontsize=12, color=COLORS["title"])
                ax_prob.axis('off')
                continue

            races = [p[0] for p in all_probs]
            probs = [p[1] for p in all_probs]
            indices = np.arange(len(races))

            # Create horizontal bars
            bars = ax_prob.barh(indices, probs, color=COLORS["bar"], height=0.7, alpha=0.85)

            # Highlight the predicted race bar
            try:
                pred_idx = races.index(predicted_race)
                bars[pred_idx].set_color(COLORS["selected_bar"])
                # Add 'Predicted' text inside the selected bar if it's wide enough
                if probs[pred_idx] > 0.15: # Threshold to prevent clutter
                     ax_prob.text(probs[pred_idx] / 2, indices[pred_idx], 'Predicted',
                                  ha='center', va='center', color=COLORS['text_light'],
                                  fontsize=8, fontweight='bold')
            except ValueError:
                logger.warning(f"Predicted race '{predicted_race}' not found in probability list for highlighting Face {face_id}.")

            # Add percentage labels next to bars
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax_prob.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                             f"{probs[j]:.1%}", va='center', ha='left',
                             color=COLORS["text_dark"], fontsize=9)

            # Formatting the probability plot
            ax_prob.set_yticks(indices)
            ax_prob.set_yticklabels([r.capitalize() for r in races], fontsize=10)
            ax_prob.set_xlim(0, 1.15) # Extra space for labels
            ax_prob.set_xlabel("Probability", fontsize=10, color=COLORS["text_dark"])
            ax_prob.set_title(f"Face {face_id} Probability Distribution", fontsize=12, color=COLORS["title"])
            ax_prob.invert_yaxis() # Display highest probability at the top

            # Style grid and spines
            ax_prob.xaxis.grid(True, linestyle='--', alpha=0.6, color=COLORS['grid'])
            ax_prob.spines['top'].set_visible(False)
            ax_prob.spines['right'].set_visible(False)
            ax_prob.spines['left'].set_linewidth(0.5)
            ax_prob.spines['bottom'].set_linewidth(0.5)
            ax_prob.tick_params(axis='both', which='major', labelsize=9)


        # --- Overall Figure Adjustments ---
        plt.suptitle(
            f"Ethnicity Prediction Analysis ({datetime.datetime.now():%Y-%m-%d %H:%M})",
            fontsize=20, y=0.99, color=COLORS["title"], fontweight="bold",
        )
        fig.text(0.5, 0.01, f"Image Source: {'Numpy Array' if isinstance(image_or_path, np.ndarray) else os.path.basename(image_or_path)}",
                 ha='center', fontsize=9, color=COLORS['text_dark'], style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent overlap

        # Save the visualization
        try:
            plt.savefig(
                output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()
            )
            logger.info(f"Prediction visualization saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}", exc_info=True)

        plt.show() # Display the plot
        plt.close(fig) # Ensure figure is closed

    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}", exc_info=True)


def predict_race(image_input, predictor_path, model_path, metadata_path="model_metadata.pkl", visualize=False, output_vis_path="prediction_visualization.png"):
    """
    Predicts the race of faces in the given image (path or array) using the trained model.
    Returns a list of prediction dictionaries.
    """
    predictions = []
    model = None
    metadata = None
    predictor = None
    detector = None

    # --- Load Model, Metadata, and Predictor ---
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return []
        model = joblib.load(model_path)

        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            logger.info(f"Loaded metadata: {metadata.get('classes', 'N/A')} classes.")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}. Cannot verify feature counts rigorously.")

        if not os.path.exists(predictor_path):
             logger.error(f"Predictor file not found: {predictor_path}")
             return []
        predictor = dlib.shape_predictor(predictor_path)
        detector = dlib.get_frontal_face_detector()

    except Exception as e:
        logger.error(f"Error loading model/predictor/metadata: {e}", exc_info=True)
        return []

    # --- Process Image and Extract Landmarks ---
    img_rgb_for_processing = None # Store the RGB image used
    try:
        # This function now returns (aligned_landmarks_flat, aligned_face_rgb, original_face_coords)
        # It handles reading the image path or array and converting to RGB internally.
        results = extract_facial_landmarks(
            image_input,
            predictor,
            detector,
            return_image=True,
            largest_only=False, # Process all detected faces
        )

        # Need the original image in RGB for visualization if input was path
        if visualize:
             if isinstance(image_input, str):
                 img_bgr = cv2.imread(image_input)
                 if img_bgr is not None:
                     img_rgb_for_processing = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                 else: # Already logged error in extract_facial_landmarks
                      pass
             elif isinstance(image_input, np.ndarray):
                  # Assume extract_facial_landmarks handled conversion if needed
                  # We need the RGB version it used. Let's try to reconstruct:
                  if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                      # Was likely BGR, convert back if needed by visualization
                      # img_rgb_for_processing = cv2.cvtColor(image_input.copy(), cv2.COLOR_BGR2RGB)
                      # --- Correction: Assume extract_landmarks worked with RGB version
                      img_rgb_for_processing = image_input.copy()
                  elif len(image_input.shape) == 2:
                      img_rgb_for_processing = cv2.cvtColor(image_input.copy(), cv2.COLOR_GRAY2RGB)
                  else: # RGBA?
                      try:
                          img_rgb_for_processing = cv2.cvtColor(image_input.copy(), cv2.COLOR_RGBA2RGB)
                      except:
                          img_rgb_for_processing = image_input.copy() # Best guess


        if not results:
            logger.info("No faces detected or processed in the image.")
            return []

        logger.info(f"Detected {len(results)} faces. Processing predictions...")

        # --- Predict for each face ---
        expected_features = -1
        if metadata and 'input_feature_count' in metadata:
            expected_features = metadata['input_feature_count']
        elif hasattr(model.named_steps.get('scaler'), 'n_features_in_'):
            expected_features = model.named_steps['scaler'].n_features_in_
        elif hasattr(model.named_steps.get('pca'), 'n_features_in_'):
             # This might be available if PCA is first, but usually after scaler
             expected_features = model.named_steps['pca'].n_features_in_


        for idx, result_tuple in enumerate(results):
            aligned_landmarks_flat, aligned_face_rgb, face_coords = result_tuple

            if aligned_landmarks_flat is None:
                logger.warning(f"Skipping face {idx} due to None landmarks.")
                continue
            if np.isnan(aligned_landmarks_flat).any():
                logger.warning(f"Skipping face {idx} due to NaN values in landmarks.")
                continue

            landmarks_reshaped = aligned_landmarks_flat.reshape(1, -1)

            # Feature count check
            if expected_features != -1 and landmarks_reshaped.shape[1] != expected_features:
                logger.error(f"Feature mismatch for face {idx}: Model expects {expected_features}, got {landmarks_reshaped.shape[1]}. Skipping prediction.")
                continue
            elif expected_features == -1:
                 logger.warning(f"Could not verify feature count for face {idx} against model.")


            try:
                # Predict probabilities using the pipeline
                pred_proba = model.predict_proba(landmarks_reshaped)[0]
                # Get class labels (races) from the model or metadata
                class_names = getattr(model, 'classes_', metadata.get('classes') if metadata else None)
                if class_names is None:
                     logger.error("Cannot determine class names from model or metadata. Skipping prediction.")
                     continue

                # Get predicted class index and name
                max_prob_idx = np.argmax(pred_proba)
                predicted_race = class_names[max_prob_idx]
                confidence = pred_proba[max_prob_idx]

                # Create map of race to probability
                race_probs = {
                    race: prob for race, prob in zip(class_names, pred_proba)
                }
                # Sort probabilities from highest to lowest for display
                sorted_probs = sorted(
                    race_probs.items(), key=lambda item: item[1], reverse=True
                )

                prediction_data = {
                    "face_id": idx,
                    "predicted_race": predicted_race,
                    "confidence": confidence,
                    "all_probabilities": sorted_probs, # List of (race, prob) tuples
                    "face_coords": face_coords,       # Original bounding box
                    "aligned_face": aligned_face_rgb, # Aligned face image (RGB)
                }
                predictions.append(prediction_data)

            except AttributeError as ae:
                 logger.error(f"Model might not support 'predict_proba' (Error: {ae}). Is it an SVC with probability=True?", exc_info=True)
                 # Optionally try model.predict if predict_proba fails
                 try:
                     predicted_race = model.predict(landmarks_reshaped)[0]
                     class_names = getattr(model, 'classes_', metadata.get('classes') if metadata else ['Unknown'])
                     prediction_data = {
                         "face_id": idx, "predicted_race": predicted_race, "confidence": 1.0, # Confidence unknown
                         "all_probabilities": [(predicted_race, 1.0)], "face_coords": face_coords,
                         "aligned_face": aligned_face_rgb,
                     }
                     predictions.append(prediction_data)
                     logger.warning(f"Used model.predict for face {idx} as predict_proba failed.")
                 except Exception as pred_err:
                     logger.error(f"Error during fallback model.predict for face {idx}: {pred_err}", exc_info=True)
                 continue # Skip if both fail
            except Exception as e:
                logger.error(f"Error during model prediction for face {idx}: {e}", exc_info=True)
                continue # Skip this face

        # --- Visualize if requested ---
        if visualize and predictions and img_rgb_for_processing is not None:
            logger.info("Generating prediction visualization...")
            visualize_predictions(img_rgb_for_processing, predictions, output_path=output_vis_path)
        elif visualize and not predictions:
             logger.info("Visualization requested but no faces were successfully predicted.")
        elif visualize and img_rgb_for_processing is None:
             logger.warning("Visualization requested but failed to load the original image for display.")


        return predictions

    except Exception as e:
        logger.error(f"Error in predict_race function: {e}", exc_info=True)
        return []


def process_camera_feed(predictor_path, model_path, metadata_path="model_metadata.pkl"):
    """
    Opens the webcam, performs real-time face detection and race prediction,
    and displays the results on the video feed.
    """
    logger.info("Initializing camera mode...")
    model = None
    metadata = None
    predictor = None
    detector = None
    class_names = None
    expected_features = -1

    # --- Load Model, Metadata, and Predictor ---
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
        model = joblib.load(model_path)

        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            class_names = metadata.get('classes')
            expected_features = metadata.get('input_feature_count', -1)
            logger.info(f"Loaded metadata: {class_names} classes, expects {expected_features} features.")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}. Trying to get info from model.")
            class_names = getattr(model, 'classes_', None)
            if hasattr(model.named_steps.get('scaler'), 'n_features_in_'):
                 expected_features = model.named_steps['scaler'].n_features_in_

        if class_names is None:
             logger.error("Could not determine class names from model or metadata.")
             return

        if not os.path.exists(predictor_path):
             logger.error(f"Predictor file not found: {predictor_path}")
             return
        predictor = dlib.shape_predictor(predictor_path)
        detector = dlib.get_frontal_face_detector()

    except Exception as e:
        logger.error(f"Failed to load model/predictor/metadata: {e}", exc_info=True)
        return

    # --- Initialize Camera ---
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    if not cap.isOpened():
        logger.error("Cannot open webcam. Check connection and permissions.")
        return

    logger.info("Camera opened successfully. Press 'q' to quit.")
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame_bgr = cap.read()  # Read frame (BGR format)
        if not ret:
            logger.error("Can't receive frame (stream end?). Exiting ...")
            break

        current_time = time.time()
        frame_count += 1

        # --- Face Detection and Prediction ---
        # Pass the BGR frame directly; extract_facial_landmarks handles conversion to RGB
        # It returns landmarks from the *aligned* face
        results = extract_facial_landmarks(
            frame_bgr, predictor, detector, return_image=True, largest_only=False
        )

        # --- Draw Results on Frame ---
        display_frame = frame_bgr.copy() # Draw on the original BGR frame

        if results:
            for idx, result_tuple in enumerate(results):
                # aligned_landmarks_flat are from the aligned face
                aligned_landmarks_flat, _, face_coords = result_tuple

                if aligned_landmarks_flat is None or np.isnan(aligned_landmarks_flat).any():
                    continue # Skip if landmarks are invalid

                landmarks_reshaped = aligned_landmarks_flat.reshape(1, -1)

                # Predict using the model pipeline
                try:
                    # Feature count check
                    if expected_features != -1 and landmarks_reshaped.shape[1] != expected_features:
                         # Log less frequently to avoid spamming console
                         if frame_count % 30 == 0: # Log every ~second
                            logger.warning(f"Feature mismatch in camera mode: Model expects {expected_features}, got {landmarks_reshaped.shape[1]}. Skipping face {idx}.")
                         continue
                    elif expected_features == -1 and frame_count % 60 == 0:
                         logger.warning("Could not verify feature count against model in camera mode.")


                    pred_proba = model.predict_proba(landmarks_reshaped)[0]
                    max_prob_idx = np.argmax(pred_proba)
                    predicted_race = class_names[max_prob_idx]
                    confidence = pred_proba[max_prob_idx]

                    # Get original bounding box coordinates from face_coords
                    x1, y1, x2, y2 = face_coords

                    # Draw bounding box (Blue)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

                    # Prepare text label
                    label = f"{predicted_race.capitalize()}: {confidence:.2f}"
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = max(y1, label_size[1] + 10) # Ensure label is below top if box is near edge

                    # Draw background rectangle for text
                    cv2.rectangle(display_frame, (x1, label_y - label_size[1] - 5),
                                  (x1 + label_size[0], label_y + base_line - 5), (255, 100, 0), cv2.FILLED)
                    # Put text (White)
                    cv2.putText(display_frame, label, (x1, label_y - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                except AttributeError as ae:
                     # Handle cases where predict_proba might not be available (log once)
                     if frame_count == 1: logger.error(f"Model may not support 'predict_proba': {ae}")
                     # Draw basic box if prediction fails
                     x1, y1, x2, y2 = face_coords
                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 1) # Red box for error
                except Exception as e:
                    # Log other prediction errors less frequently
                    if frame_count % 30 == 0:
                        logger.error(f"Error during prediction/drawing in camera mode: {e}")
                    # Draw basic box if prediction fails
                    x1, y1, x2, y2 = face_coords
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 1) # Red box for error


        # --- Calculate and Display FPS ---
        # Calculate FPS over a window (e.g., every second) for stability
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            prev_time = current_time
            frame_count = 0

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Display the resulting frame ---
        try:
            cv2.imshow("Real-time Ethnicity Detection (Press 'q' to quit)", display_frame)
        except cv2.error as cv_err:
             # Handle potential window errors (e.g., window closed unexpectedly)
             if "NULL window" in str(cv_err):
                 logger.warning("Display window seems to have been closed.")
                 break
             else:
                 logger.error(f"OpenCV display error: {cv_err}")
                 break


        # --- Exit Condition ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("Quit key pressed. Exiting camera mode.")
            break
        # Add check for window close button press (may vary by OS/backend)
        try:
             if cv2.getWindowProperty("Real-time Ethnicity Detection (Press 'q' to quit)", cv2.WND_PROP_VISIBLE) < 1:
                 logger.info("Window closed by user. Exiting camera mode.")
                 break
        except cv2.error: # Handle case where window doesn't exist yet or anymore
             pass


    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    # Add a small delay to ensure windows close properly on some systems
    cv2.waitKey(1)
    logger.info("Camera released and windows closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced Ethnicity Detection Model: Train, Predict, or Camera Feed Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "camera"],
        required=True,
        help="Operational mode: 'train' a new model, 'predict' on a single image, or run 'camera' feed.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./training_data", # Example default
        help="Directory containing subdirectories of images for each race/ethnicity class (required for 'train' mode).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the input image file for prediction (required for 'predict' mode).",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib's facial landmark predictor file (.dat).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ethnicity_detection_model.pkl",
        help="Path to save the trained model (in 'train' mode) or load it from (in 'predict'/'camera' modes).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="model_metadata.pkl",
        help="Path to save/load model metadata (classes, parameters, etc.). Complements the main model file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and display/save detailed visualization of predictions (only for 'predict' mode).",
    )
    parser.add_argument(
        "--output_vis",
        type=str,
        default="prediction_visualization.png",
        help="Output path for the prediction visualization image (if --visualize is used in 'predict' mode)."
    )

    # Group for augmentation arguments
    aug_group = parser.add_argument_group('Augmentation Options (for training mode)')
    aug_group.add_argument(
        "--augmentation",
        action="store_true",
        # default=True, # Let the absence of --no-augmentation imply True
        help="Enable data augmentation during training. Enabled by default unless --no-augmentation is specified.",
    )
    aug_group.add_argument(
        "--no-augmentation",
        action="store_false",
        dest="augmentation", # Make --no-augmentation set augmentation to False
        help="Disable data augmentation during training.",
    )
    # Set default for augmentation based on the flags
    parser.set_defaults(augmentation=True)

    aug_group.add_argument(
        "--aug_factor",
        type=int,
        default=3, # Increased default augmentation factor
        help="Number of augmented samples to generate per original image during training.",
    )
    args = parser.parse_args()

    # --- Validate common arguments ---
    if not os.path.exists(args.predictor):
        logger.error(
            f"Predictor file not found: {args.predictor}. "
            "Please download it (e.g., from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it correctly."
        )
        exit(1)

    # --- Mode Execution ---
    if args.mode == "train":
        logger.info("--- Starting Training Mode ---")
        if not args.data_dir:
            logger.error("Training mode requires the --data_dir argument specifying the image dataset location.")
            exit(1)
        if not os.path.isdir(args.data_dir):
            logger.error(f"Data directory not found or is not a directory: {args.data_dir}")
            exit(1)

        logger.info(f"Using data directory: {args.data_dir}")
        logger.info(f"Data augmentation enabled: {args.augmentation}")
        if args.augmentation:
            logger.info(f"Augmentation factor: {args.aug_factor}")
        logger.info(f"Model will be saved to: {args.model}")
        logger.info(f"Metadata will be saved to: {args.metadata}")

        logger.info("Step 1: Creating training dataset...")
        X, y = create_training_dataset(
            args.data_dir,
            args.predictor,
            augmentation=args.augmentation,
            augmentation_factor=args.aug_factor,
        )
        if not X or len(X) == 0:
            logger.error("Dataset creation failed or resulted in zero samples. Cannot proceed with training. Check logs.")
            exit(1)

        logger.info("Step 2: Training model...")
        # Pass save paths to train_model
        model, X_test, y_test = train_model(X, y, model_save_path=args.model, metadata_save_path=args.metadata)

        if model is None:
             logger.error("Model training failed. Check logs for details. Exiting.")
             exit(1)

        logger.info("--- Model training complete! ---")

    elif args.mode == "predict":
        logger.info("--- Starting Prediction Mode ---")
        if not args.image_path:
            logger.error("Prediction mode requires the --image_path argument.")
            exit(1)
        if not os.path.exists(args.image_path):
            logger.error(f"Input image file not found: {args.image_path}")
            exit(1)
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}. Cannot perform prediction.")
            exit(1)
        # Metadata file is optional but recommended for predict
        if not os.path.exists(args.metadata):
             logger.warning(f"Metadata file not found: {args.metadata}. Prediction will proceed but feature count checks might be limited.")


        logger.info(f"Loading model from: {args.model}")
        logger.info(f"Predicting on image: {args.image_path}")
        logger.info(f"Visualization enabled: {args.visualize}")
        if args.visualize:
             logger.info(f"Visualization output path: {args.output_vis}")


        predictions = predict_race(
            args.image_path,
            args.predictor,
            args.model,
            metadata_path=args.metadata, # Pass metadata path
            visualize=args.visualize,
            output_vis_path=args.output_vis, # Pass vis output path
        )

        if not predictions:
            print("\nNo faces were detected or successfully processed in the image.")
        else:
            print(f"\n--- Prediction Results ({len(predictions)} faces found) ---")
            for pred in predictions:
                print(f"\nFace {pred['face_id']} (Coords: {pred['face_coords']}):")
                print(f"  Predicted Ethnicity: {pred['predicted_race'].capitalize()}")
                print(f"  Confidence: {pred['confidence']:.3f}")
                print("  Top Probabilities:")
                # Show top 3 or all if fewer than 3 classes
                num_probs_to_show = min(3, len(pred['all_probabilities']))
                for race, prob in pred["all_probabilities"][:num_probs_to_show]:
                    print(f"    - {race.capitalize()}: {prob:.3f}")
        logger.info("--- Prediction complete! ---")

    elif args.mode == "camera":
        logger.info("--- Starting Camera Mode ---")
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}. Cannot run camera mode.")
            exit(1)
        # Metadata file is optional but recommended for camera
        if not os.path.exists(args.metadata):
             logger.warning(f"Metadata file not found: {args.metadata}. Camera mode will proceed but feature count checks might be limited.")

        logger.info(f"Loading model from: {args.model}")
        process_camera_feed(args.predictor, args.model, metadata_path=args.metadata)
        logger.info("--- Camera mode finished. ---")

    else:
        # This case should not be reachable due to 'choices' in argparse
        logger.error(f"Invalid mode '{args.mode}' selected. Use 'train', 'predict', or 'camera'.")
        exit(1)\
