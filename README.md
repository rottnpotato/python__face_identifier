## Installation

1. Python 3.7+
2. Clone the repository:
```bash
git clone https://github.com/yourusername/facial-race-detection.git
cd facial-race-detection
```
 or download it as zip.
3. if downloaded as zip, extract files.
4. navigate to the root project directory.
5. install CMake first (search it on google).
6. install required packages:
```bash
pip install numpy opencv-python dlib scikit-image scikit-learn imutils matplotlib tqdm joblib
```
7. dowload this file (`http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`) and extract it, copy the `shape_predictor_68_face_landmarks.dat` to your projects directory (along side training_data).

## Each subdirectory name will be used as the race label.
```
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
```

Run the training script:

```bash
python training.py
```


This will:
1. Process all images in the training_data directory
2. Extract and align faces
3. Apply data augmentation
4. Train an SVM model with hyperparameter tuning
5. Save the trained model as `race_detection_model.pkl`
6. Generate evaluation metrics and visualizations

## Prediction

To predict race from a new image:

```bash
python prediction.py path/to/image.jpg --visualize
```

Options:
- `--predictor`: Path to the facial landmark predictor (default: shape_predictor_68_face_landmarks.dat)
- `--model`: Path to the trained model (default: race_detection_model.pkl)
- `--visualize`: Generate visualization of the predictions






