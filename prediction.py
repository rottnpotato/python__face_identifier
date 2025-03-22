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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("prediction.log"), logging.StreamHandler()],
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
        # Handle both file paths and already loaded images
        if isinstance(image_path, str):
            img = io.imread(image_path)
            if img.shape[2] == 4:  # Handle RGBA images
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = image_path
            
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            logger.warning("No faces detected in the image")
            return None if not return_image else (None, None, None)

        # Process all faces and return a list of results
        results = []
        
        for face_idx, face in enumerate(faces):
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Align the face
            aligned_face = align_face(img, landmarks, desired_face_width)
            if aligned_face is None:
                continue

            # Convert the aligned face to grayscale
            aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)

            # Histogram equalization
            aligned_gray = cv2.equalizeHist(aligned_gray)

            # Re-detect landmarks on the aligned face
            face_rect = dlib.rectangle(0, 0, desired_face_width, desired_face_width)
            aligned_landmarks = predictor(aligned_gray, face_rect)
            aligned_landmarks = face_utils.shape_to_np(aligned_landmarks)
            
            # Store the original face coordinates for visualization
            face_coords = (face.left(), face.top(), face.right(), face.bottom())
            
            if return_image:
                results.append((aligned_landmarks.flatten(), aligned_face, face_coords))
            else:
                results.append((aligned_landmarks.flatten(), face_coords))

        return results

    except Exception as e:
        logger.error(f"Error extracting landmarks: {e}")
        return None


def predict_race(image_path, predictor_path, model_path, visualize=False):
    """
    Predicts the race of people in an image based on facial landmarks.
    Returns a list of predictions with confidence scores.
    """
    try:
        # Load the trained model and metadata
        model = joblib.load(model_path)
        metadata = joblib.load("model_metadata.pkl")
        
        # Load the shape predictor
        predictor = dlib.shape_predictor(predictor_path)
        detector = dlib.get_frontal_face_detector()
        
        # Extract facial landmarks from the image
        results = extract_facial_landmarks(
            image_path, predictor, detector, return_image=visualize
        )
        
        if not results:
            logger.warning("No faces detected or processed in the image.")
            return []
        
        predictions = []
        
        # Process each face
        for idx, result in enumerate(results):
            if visualize:
                landmarks, aligned_face, face_coords = result
            else:
                landmarks, face_coords = result
                
            if landmarks is None:
                continue
                
            # Reshape landmarks to a 2D array
            landmarks = landmarks.reshape(1, -1)
            
            # Predict the race and the probabilities
            predicted_race = model.predict(landmarks)[0]
            probabilities = model.predict_proba(landmarks)[0]
            
            # Get all probabilities with class labels
            race_probs = {
                race: prob
                for race, prob in zip(model.classes_, probabilities)
            }
            
            # Sort probabilities
            sorted_probs = sorted(
                race_probs.items(), key=lambda x: x[1], reverse=True
            )
            
            prediction = {
                "face_id": idx,
                "predicted_race": predicted_race,
                "confidence": race_probs[predicted_race],
                "all_probabilities": sorted_probs,
                "face_coords": face_coords
            }
            
            if visualize:
                prediction["aligned_face"] = aligned_face
                
            predictions.append(prediction)
            
        # Visualize results if requested
        if visualize and predictions:
            visualize_predictions(image_path, predictions)
            
        return predictions
        
    except Exception as e:
        logger.error(f"Error predicting race: {e}")
        return []


def visualize_predictions(image_path, predictions):
    """
    Visualizes the predictions on the original image.
    """
    try:
        # Load the original image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path.copy()
            
        # Create a figure with subplots
        n_faces = len(predictions)
        fig = plt.figure(figsize=(15, 5 * (n_faces + 1)))
        
        # Show the original image with bounding boxes
        ax_main = plt.subplot(n_faces + 1, 2, 1)
        ax_main.imshow(img)
        ax_main.set_title("Original Image with Detections")
        ax_main.axis("off")
        
        # Draw bounding boxes and labels
        for pred in predictions:
            x1, y1, x2, y2 = pred["face_coords"]
            race = pred["predicted_race"]
            conf = pred["confidence"]
            
            # Draw rectangle
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor="r", facecolor="none"
            )
            ax_main.add_patch(rect)
            
            # Add label
            ax_main.text(
                x1, y1 - 10, 
                f"Face {pred['face_id']}: {race} ({conf:.2f})",
                color="white", fontsize=12, 
                bbox=dict(facecolor="red", alpha=0.7)
            )
        
        # Show each aligned face with its prediction
        for i, pred in enumerate(predictions):
            # Show aligned face
            ax_face = plt.subplot(n_faces + 1, 2, 3 + i * 2)
            ax_face.imshow(pred["aligned_face"])
            ax_face.set_title(f"Face {pred['face_id']} (Aligned)")
            ax_face.axis("off")
            
            # Show probability bar chart
            ax_prob = plt.subplot(n_faces + 1, 2, 4 + i * 2)
            
            races = [p[0] for p in pred["all_probabilities"]]
            probs = [p[1] for p in pred["all_probabilities"]]
            
            ax_prob.barh(races, probs, color="skyblue")
            ax_prob.set_xlim(0, 1)
            ax_prob.set_title(f"Probability Distribution for Face {pred['face_id']}")
            ax_prob.set_xlabel("Probability")
            
            # Highlight the predicted race
            idx = races.index(pred["predicted_race"])
            ax_prob.get_children()[idx].set_color("red")
        
        plt.tight_layout()
        plt.savefig("prediction_visualization.png")
        plt.show()
        
    except Exception as e:
        logger.error(f"Error visualizing predictions: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict race from facial image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--predictor", 
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to the facial landmark predictor"
    )
    parser.add_argument(
        "--model", 
        default="race_detection_model.pkl",
        help="Path to the trained race detection model"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize the predictions"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        exit(1)
        
    if not os.path.exists(args.predictor):
        logger.error(
            f"Predictor file not found: {args.predictor}. "
            "Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )
        exit(1)
        
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        exit(1)
    
    # Predict race
    predictions = predict_race(
        args.image_path, args.predictor, args.model, args.visualize
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
            for race, prob in pred['all_probabilities']:
                print(f"    {race}: {prob:.4f}")
