A collection of projects completed during the DecodeLabs AI Engineer training program, covering rule-based systems, supervised learning, recommendation engines, and computer vision.

## Task1: Rule-Based AI Chatbot (MHTBot)

A simple chatbot built using pattern matching and conditional logic, no machine learning involved. Handles greetings, identity questions, general AI/ML definitions, time/date queries, and jokes through a Tkinter desktop GUI.

**Tech stack:** Python, Tkinter

**Run it:**
```
python mhtbot_gui.py
```

## Task2: Data Classification Using AI (Iris Classifier)

A supervised learning pipeline built on the classic Iris dataset. Covers the full ML workflow: feature scaling, train-test split, KNN classification, and model evaluation using accuracy, F1 score, and a confusion matrix. Includes a Tkinter GUI for live predictions.

**Tech stack:** Python, scikit-learn, Tkinter, NumPy

**Run it:**
```
pip install scikit-learn numpy
python iris_classifier_gui.py
```

## Task3: AI Recommendation Logic (Tech Stack Recommender)

A content-based recommendation engine that maps a user's skills to relevant career paths. Uses TF-IDF vectorization to weight skill relevance and Cosine Similarity to rank job roles, returning the Top 3 matches.

**Tech stack:** Python, scikit-learn, Tkinter

**Run it:**
```
pip install scikit-learn
python tech_stack_recommender_gui.py
```

## Task4: Image and Text Recognition (OCR + Object Detection)

A computer vision pipeline supporting two paths: OCR text extraction using Tesseract, and object detection using a pre-trained MobileNet-SSD model. Includes image preprocessing (grayscale, Gaussian blur, Otsu thresholding) and an 80% confidence gate to filter out low-confidence predictions.

**Tech stack:** Python, OpenCV, pytesseract, Tkinter

**Setup:**
1. Install Tesseract OCR for Windows: https://github.com/UB-Mannheim/tesseract/wiki
2. Install Python dependencies:
```
pip install opencv-python numpy pytesseract
```
3. Make sure `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` are in the same folder as the script.

**Run it:**
```
python vision_recognition_gui.py
```
