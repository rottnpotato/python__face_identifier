## Prerequisites

1. Python 3.7+
2. Required packages:

```pip install numpy opencv-python dlib scikit-image scikit-learn imutils matplotlib tqdm joblib```

=============================================================================================
## Usage steps:

1. navigate to the projects directory
2. follow the commands below.


## Each subdirectory name will be used as the race label.

training_data/
├── asian/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── black/
│   ├── image1.jpg
│   └── ...
├── white/
│   ├── image1.jpg
│   └── ...
└── ...


Run the training script:

```python training.py```


This will:
1. Process all images in the training_data directory
2. Extract and align faces
3. Apply data augmentation
4. Train an SVM model with hyperparameter tuning
5. Save the trained model as `race_detection_model.pkl`
6. Generate evaluation metrics and visualizations

## Prediction

To predict race from a new image:

```python prediction.py path/to/image.jpg --visualize```

Options:
- `--predictor`: Path to the facial landmark predictor (default: shape_predictor_68_face_landmarks.dat)
- `--model`: Path to the trained model (default: race_detection_model.pkl)
- `--visualize`: Generate visualization of the predictions






