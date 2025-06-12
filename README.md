This project is a real-time sign language detector that uses your webcam and deep learning to recognize simple hand gestures such 
as "hello", "yes",, "thanks", "mom", "dad", and "no" using MediaPipe and TensorFlow.

## Features
- Real-time hand detection via webcam  
- Trained with custom data for three signs: **hello**, **yes**, and **no**  
- Visual overlay showing detected gesture label  
- Easily extendable to include more signs

## Tech Stack
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- scikit-learn

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sign-language-detector.git
cd sign-language-detector
```

### 2. Install Dependencies
Install required packages using pip:
```bash
pip install -r requirements.txt
```

### 3. Collect Training Data
Use your webcam to collect hand sign data:

```bash
python collect_data.py
```
Press the following keys while showing the corresponding sign:

- h → hello
- y → yes
- n → no
- t → thanks
- m → mom
- d → dad

### 4. Train the Model
Train a neural network on your collected data:

```bash
python train_model.py
```
### 5. Predict in Real Time
Use the trained model to predict signs via webcam:
```bash
python predict_sign.py
```

### 6. To Exit
Press q in any window to quit the app.

