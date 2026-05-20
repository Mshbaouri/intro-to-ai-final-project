# 🔐 AI Smart Lock System

> **Introduction to Artificial Intelligence — Final Project**
> Egypt University of Informatics

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [AI & Algorithms](#ai--algorithms)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Team](#team)

---

## Overview

The **AI Smart Lock System** is a Python-based intelligent door lock that uses facial recognition and liveness detection to grant or deny access to registered users. Instead of keys or PIN codes, the system identifies family members in real time through a webcam, verifies they are physically present (not a photo or video), and sends an unlock or lock signal to a connected microprocessor accordingly.

Unauthorized visitors trigger an automatic alert — including a saved snapshot and a notification email — while anomaly detection flags suspicious behaviour such as repeated failed attempts or access during restricted night-time hours.

---

## Features

- **Face Recognition** — Identifies registered family members using Google FaceNet embeddings and a K-Nearest Neighbours (KNN) classifier.
- **Liveness Detection** — Randomised challenge at every unlock attempt (blink detection via Eye Aspect Ratio *or* head-turn challenge) to prevent spoofing with photos or videos.
- **Auto-Learning** — Automatically saves new embeddings of recognised members over time for improved accuracy.
- **Anomaly Detection** — Raises alerts for repeated stranger detections (brute-force attempts) and night-time entry events.
- **Intruder Snapshots** — Captures and stores a photo of any unrecognised visitor, then sends an email alert with the image attached.
- **Access Logging** — All events (granted, denied, liveness failure, registration, auto-learn) are recorded to a timestamped CSV log.
- **Flexible Registration** — Supports webcam capture, bulk image uploads, and group/multi-face photo import.

---

## AI & Algorithms

| Component | Technique |
|---|---|
| Face embedding | Google FaceNet (via `keras-facenet`) — 512-dimensional embeddings |
| Face detection | Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) |
| Identity classification | K-Nearest Neighbours (KNN) with majority voting |
| Liveness — blink | Eye Aspect Ratio (EAR) on dlib 68-point facial landmarks |
| Liveness — head turn | Nose-tip yaw ratio from dlib 68-point facial landmarks |
| Distance metric | L2 (Euclidean) norm |

---

## Project Structure

```
intro-to-ai-final-project/
│
├── smart-lock-code.py          # Main application
│
└── SmartLock/                  # Auto-created at runtime
    ├── face_database/          # Per-person folders with .jpg + .npy embeddings
    │   └── <Name>/
    ├── access_log.csv          # Timestamped event log
    ├── upload_queue/           # Drop individual photos here for bulk import
    └── mixed_uploads/          # Drop group photos here for multi-face import
```

> The `SmartLock/` directory and all sub-folders are created automatically on first run.

---

## Requirements

- Python 3.8+
- Webcam
- [`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) — download, extract, and place in the same directory as `smart-lock-code.py`

### Python Dependencies

```
opencv-python
numpy
tensorflow
keras-facenet
scipy
dlib  (or dlib-bin for easier Windows install)
```

Install all at once:

```bash
pip install opencv-python numpy tensorflow keras-facenet scipy dlib-bin
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Mshbaouri/intro-to-ai-final-project.git
cd intro-to-ai-final-project
```

2. **Install dependencies**

```bash
pip install opencv-python numpy tensorflow keras-facenet scipy dlib-bin
```

3. **Download the dlib landmark model**

Download [`shape_predictor_68_face_landmarks.dat.bz2`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place the `.dat` file in the project root.

4. **Run the application**

```bash
python smart-lock-code.py
```

---

## Usage

### Registering a Family Member (Webcam)

```python
register_family_member("Name", num_photos=3)
```

Press **SPACE** to capture each photo, **ESC** to skip.

### Registering from Uploaded Images

Place individual photos inside `SmartLock/upload_queue/<Name>/` then call:

```python
register_from_files()
```

### Attempting to Unlock

```python
attempt_unlock()
```

Press **SPACE** to scan your face. If recognised, a randomised liveness challenge (blink or head-turn) is presented before the unlock signal is sent.

### Live Continuous Recognition

```python
run_live_recognition()
```

Runs a continuous recognition loop on the webcam feed.

### Evaluating Model Accuracy

```python
evaluate_knn(test_folder="path/to/test/images", threshold=0.6, k=3)
```

---

## Team

| Name | 
|---|
| Mohamed Ezzat |
| Maryam Rageh |
| Mariam Hassan |
| Abdelrahman ElKhashab |

**Course:** Introduction to Artificial Intelligence
**Institution:** Egypt University of Informatics
