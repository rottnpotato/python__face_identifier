## Installation
```text
1. Python 3.7+
2. Clone the repository:
```
```bash
git clone https://github.com/rottnpotato/python__face_identifier.git
cd python__face_identifier
```
```text
 or download it as zip.
3. if downloaded as zip, extract files.
4. navigate to the root project directory.
5. install visual studio build tools, then select desktop development and select c++ for cmake then install.
6. install CMake (search it on google).
7. install required packages:
```
```bash
pip install numpy opencv-python dlib scikit-image scikit-learn imutils matplotlib tqdm joblib
```
8. dowload this file (`http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`) and extract it, copy the `shape_predictor_68_face_landmarks.dat` to your projects directory (along side training_data).

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
python images.py --mode train --data_dir training_data --augmentation --aug_factor 15 
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
python images.py --mode predict path/to/image.jpg --visualize
```

## NOTE: Some results are inaccurate, maybe use more data for the model to train upon, you can add more race or change some images or train it again.
## NOTE: Inside the training_data folder you'll see subfolder, those are the race names and the images inside it are training data.
## Explore the code and modify it to your liking. 


