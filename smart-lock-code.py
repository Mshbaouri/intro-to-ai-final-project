import os
import sys
import winsound
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import csv
import numpy as np
from datetime import datetime
import tensorflow as tf
from scipy.spatial import distance as scipy_dist
from collections import Counter

tf.get_logger().setLevel("ERROR")
from keras_facenet import FaceNet

# dlib for facial landmark detection (liveness / head-turn check)
try:
    import dlib

    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("[WARN] dlib not found - advanced liveness detection disabled.")
    print("       Run:  pip install dlib-bin  to enable head-turn detection.\n")

# ==============================================================================
#  SETUP
# ==============================================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

TESTING_NIGHT_MODE = True  # Turn this to False when you're done testing the restricted entry timing

DATABASE_PATH = './SmartLock/face_database/'
LOG_PATH = './SmartLock/access_log.csv'

os.makedirs(DATABASE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

print("Loading Google FaceNet AI...")
embedder = FaceNet()
print("[OK] AI Loaded and Ready!\n")

AUTHORIZED_FAMILY: dict[str, list[np.ndarray]] = {}

# ==============================================================================
#  LIVENESS DETECTION — Eye Aspect Ratio (EAR) for Blink Detection
# ==============================================================================
_LEFT_EYE_IDX = list(range(42, 48))  # dlib 68-point model: left eye
_RIGHT_EYE_IDX = list(range(36, 42))  # dlib 68-point model: right eye

EAR_BLINK_THRESHOLD = 0.25  # EAR below this → eye is closed
EAR_CONSEC_FRAMES = 2  # consecutive closed frames needed to count a blink
BLINKS_REQUIRED = 1  # blinks required to pass liveness

def _eye_aspect_ratio(eye_pts):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
    A = scipy_dist.euclidean(eye_pts[1], eye_pts[5])
    B = scipy_dist.euclidean(eye_pts[2], eye_pts[4])
    C = scipy_dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)

def check_liveness_blink(timeout_seconds: float = 10.0, cap=None) -> bool:
    """
    Checks for a valid blink using Eye Aspect Ratio (EAR) on a continuous video feed.
    """
    import time
    detector, predictor = _get_dlib_models()
    if detector is None:
        print("[LIVENESS] Skipping blink check (dlib model missing).")
        return True

    # Use existing camera feed to prevent "gaps", otherwise open a new one
    release_cap = False
    if cap is None:
        cap = cv2.VideoCapture(0)
        release_cap = True

    print("\n[LIVENESS] Blink challenge started. Please BLINK facing the camera.")

    blink_counter = 0
    blinks_total = 0
    passed = False
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            print("[LIVENESS] Timed out - blink challenge failed.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

            # Extract eyes and calculate EAR
            left_eye = landmarks[_LEFT_EYE_IDX]
            right_eye = landmarks[_RIGHT_EYE_IDX]

            left_ear = _eye_aspect_ratio(left_eye)
            right_ear = _eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Draw green outlines around the eyes for visual feedback
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            # Blink logic
            if ear < EAR_BLINK_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    blinks_total += 1
                blink_counter = 0

            cv2.putText(frame, f"Blinks: {blinks_total} / {BLINKS_REQUIRED}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if blinks_total >= BLINKS_REQUIRED:
                passed = True
                break

        cv2.putText(frame, f"Time: {max(0, timeout_seconds - elapsed):.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255), 2)
        cv2.imshow("Smart Lock - Liveness (Please Blink)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        if passed:
            print("[LIVENESS] PASSED - Live person confirmed (blinked).")
            cv2.putText(frame, "PASSED!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Smart Lock - Liveness (Please Blink)", frame)
            cv2.waitKey(800)
            break

    if release_cap:
        cap.release()
    cv2.destroyAllWindows()
    return passed

# ==============================================================================
#  LIVENESS DETECTION (Head-Turn Challenge)
# ==============================================================================
_NOSE_TIP_IDX = 30  # dlib landmark index for the nose tip
_LEFT_THRESH = 0.40  # nose must cross this left (normalised x < this)
_RIGHT_THRESH = 0.60  # nose must cross this right (normalised x > this)

_dlib_detector = None
_dlib_predictor = None

def _get_dlib_models():
    """Lazy-load dlib's face detector and 68-point landmark predictor."""
    global _dlib_detector, _dlib_predictor
    if not DLIB_AVAILABLE:
        return None, None
    if _dlib_detector is None:
        _dlib_detector = dlib.get_frontal_face_detector()
        # The shape predictor file must be in the same folder as this script.
        # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.isfile(predictor_path):
            print(f"\n[LIVENESS] WARNING: '{predictor_path}' not found.")
            print("  Liveness detection requires this file.")
            print("  Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("  Extract and place in the same folder as this script.\n")
            return None, None
        _dlib_predictor = dlib.shape_predictor(predictor_path)
    return _dlib_detector, _dlib_predictor

def check_liveness_head_turn(target_side="LEFT", timeout_seconds: float = 10.0, cap=None) -> bool:
    """
    Asks the user to turn their head to ONE specific side (LEFT or RIGHT).
    Returns True immediately when the target is hit.
    """
    import time

    # Ensure these match your global constants
    _NOSE_TIP_IDX = 33
    _LEFT_THRESH = 0.35  # Adjust based on your previous calibration
    _RIGHT_THRESH = 0.65  # Adjust based on your previous calibration

    detector, predictor = _get_dlib_models()
    if detector is None:
        print("[LIVENESS] Skipping head-turn check (dlib not available).")
        return True

    release_cap = False
    if cap is None:
        cap = cv2.VideoCapture(0)
        release_cap = True

    print(f"\n[LIVENESS] Head-turn challenge started: PLEASE TURN {target_side}")

    start_time = time.time()
    passed = False
    nose_norm = 0.5  # Default center

    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            print(f"[LIVENESS] Timed out - failed to turn {target_side}.")
            break

        ret, frame = cap.read()
        if not ret: break

        # Mirror the frame if needed so 'Left' feels like 'Left' to the user
        # frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        face_found = False
        for rect in rects:
            face_found = True
            shape = predictor(gray, rect)
            landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

            # Logic to calculate where the nose is relative to face width
            face_left = rect.left()
            face_right = rect.right()
            face_w = max(face_right - face_left, 1)
            nose_x = landmarks[_NOSE_TIP_IDX][0]

            # This is your 'Yaw Ratio'
            nose_norm = (nose_x - face_left) / face_w

            # Draw visual feedback
            cx, cy = int(landmarks[_NOSE_TIP_IDX][0]), int(landmarks[_NOSE_TIP_IDX][1])
            cv2.circle(frame, (cx, cy), 5, (0, 200, 255), -1)
            cv2.rectangle(frame, (face_left, rect.top()), (face_right, rect.bottom()), (0, 200, 255), 1)

            # --- SINGLE DIRECTION CHECK ---
            if target_side == "LEFT" and nose_norm < _LEFT_THRESH:
                passed = True
            elif target_side == "RIGHT" and nose_norm > _RIGHT_THRESH:
                passed = True

            if passed: break

        if passed:
            # Short delay so the user sees the 'PASSED' message
            cv2.putText(frame, "PASSED!", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 0), 3)
            cv2.imshow("Smart Lock - Liveness", frame)
            cv2.waitKey(800)
            break

        # UI OVERLAY
        remaining = max(0, timeout_seconds - elapsed)
        instruction = f"PLEASE TURN {target_side} ->" if target_side == "RIGHT" else f"<- PLEASE TURN {target_side}"

        # Draw the progress bar at the top
        bar_w, bar_x0, bar_y, bar_h = 300, 10, 85, 18
        fill = int(nose_norm * bar_w)
        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + fill, bar_y + bar_h), (0, 200, 255), -1)

        cv2.putText(frame, f"Challenge: {target_side} | Time: {remaining:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, instruction, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        if not face_found:
            cv2.putText(frame, "No face detected", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Smart Lock - Liveness", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    if release_cap:
        cap.release()
    cv2.destroyAllWindows()
    return passed

''''
# ==============================================================================
#  PASSIVE LIVENESS: Texture Analysis (LBP)
# ==============================================================================
def check_texture_lbp(frame: np.ndarray,
                      threshold: float = 6.5,
                      face_padding: float = 0.15) -> tuple:
    """
    Passive liveness check using Local Binary Patterns (LBP) texture analysis.
    Calculates the 'Entropy' (randomness) of micro-textures.
    Real skin = high entropy. Phone screens/paper = lower entropy.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    if len(faces) == 0:
        print("[TEXTURE] No face detected for texture check — skipping.")
        return True, 0.0

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Crop inward to focus purely on the cheek/nose skin, avoiding the background
    pad_x = int(w * face_padding)
    pad_y = int(h * face_padding)
    face_roi = gray[y + pad_y: y + h - pad_y, x + pad_x: x + w - pad_x]

    if face_roi.size == 0:
        return True, 0.0

    # --- Pure NumPy Vectorized LBP (Super Fast) ---
    im_pad = np.pad(face_roi, ((1, 1), (1, 1)), mode='edge')
    center = im_pad[1:-1, 1:-1]

    lbp = np.zeros_like(center, dtype=np.uint8)

    lbp |= ((im_pad[0:-2, 0:-2] >= center) << 7).astype(np.uint8)
    lbp |= ((im_pad[0:-2, 1:-1] >= center) << 6).astype(np.uint8)
    lbp |= ((im_pad[0:-2, 2:] >= center) << 5).astype(np.uint8)
    lbp |= ((im_pad[1:-1, 2:] >= center) << 4).astype(np.uint8)
    lbp |= ((im_pad[2:, 2:] >= center) << 3).astype(np.uint8)
    lbp |= ((im_pad[2:, 1:-1] >= center) << 2).astype(np.uint8)
    lbp |= ((im_pad[2:, 0:-2] >= center) << 1).astype(np.uint8)
    lbp |= ((im_pad[1:-1, 0:-2] >= center) << 0).astype(np.uint8)

    # --- Calculate LBP Histogram & Entropy ---
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    entropy = -np.sum(hist * np.log2(hist + 1e-7))

    passed = entropy >= threshold
    status = "PASS" if passed else "FAIL"
    print(f"[TEXTURE] LBP Entropy={entropy:.2f}  threshold={threshold}  [{status}]")

    return passed, entropy

'''
# ==============================================================================
#  ANOMALY DETECTION CONFIGURATION
# ==============================================================================
RESTRICTED_START = 0
RESTRICTED_END = 5
GLOBAL_THRESHOLD = 0.60
BRUTE_FORCE_LIMIT = 10
COOLDOWN_WINDOW = 30
ALERT_COOLDOWN = 10
last_alert_time = 0
stranger_counter = 0
last_stranger_time = 0

def play_alarm(type="warning"):
    if type == "danger":
        winsound.Beep(1500, 600)
    elif type == "info":
        winsound.Beep(800, 200)

def check_for_anomalies(name, distance):
    global stranger_counter, last_stranger_time, last_alert_time
    import time

    now_ts = time.time()
    now = datetime.now()

    if TESTING_NIGHT_MODE:
        now = now.replace(hour=3, minute=15)

    current_hour = now.hour
    current_time_str = now.strftime("%I:%M:%S %p")
    is_night = RESTRICTED_START <= current_hour < RESTRICTED_END

    match_percent = max(0, min(100, (1 - distance) * 100))

    alert_msg = ""
    sound_type = None

    if name == "STRANGER":
        if (now_ts - last_stranger_time) > 1.0:
            stranger_counter += 1
            last_stranger_time = now_ts

            if stranger_counter >= BRUTE_FORCE_LIMIT and stranger_counter % 10 == 0:
                if is_night:
                    alert_msg = f"HIGH-RISK: Unknown visitor at {current_time_str}!"
                    sound_type = "danger"
                else:
                    alert_msg = f"VISITOR ALERT: Unknown person present for {stranger_counter}s!"
                    sound_type = "danger"

        if (now_ts - last_stranger_time) > COOLDOWN_WINDOW:
            stranger_counter = 0
    else:
        if is_night:
            if (now_ts - last_alert_time) > ALERT_COOLDOWN:
                alert_msg = f"NIGHT ENTRY: {name} at {current_time_str} ({match_percent:.1f}% Match)."
                sound_type = "info"

        stranger_counter = 0
        last_stranger_time = now_ts

    if alert_msg and (now_ts - last_alert_time) > ALERT_COOLDOWN:
        print("\n" + "!" * 60)
        print(alert_msg)
        print("!" * 60 + "\n")

        if sound_type:
            play_alarm(sound_type)
        send_alert(
            subject=f"[Smart Lock] {alert_msg}",
            body=(
                f"Smart Lock Alert\n"
                f"─────────────────\n"
                f"{alert_msg}\n\n"
                f"Action : Check your door camera.\n"
            )
        )
        last_alert_time = now_ts

# ==============================================================================
#  CORE HELPERS
# ==============================================================================
def preprocess_for_facenet(raw_img: np.ndarray):
    """Detect the first face in a BGR webcam frame and return a (1,160,160,3)
    RGB batch ready for FaceNet. Returns None if no face is found."""
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = cv2.resize(raw_img[y:y + h, x:x + w], (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return np.expand_dims(face, axis=0)

def get_embedding(img: np.ndarray):
    """Return a (1, 512) FaceNet embedding or None if no face detected."""
    processed = preprocess_for_facenet(img)
    if processed is not None:
        return embedder.embeddings(processed)
    return None

def take_photo(window_title: str = "Camera - SPACE = capture | ESC = cancel"):
    """Open the default webcam, show a live preview, and return the captured
    frame on SPACE or None on ESC."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None

    print(f"  [Camera] SPACE = capture  |  ESC = cancel")
    img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  [Camera] Failed to grab frame.")
            break

        cv2.imshow(window_title, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("  [Camera] Cancelled.")
            break
        elif key == 32:
            print("  [Camera] [CAPTURE] Captured!")
            img = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return img

def log_event(event: str, person_name: str, details: str = ""):
    """Append one row to the CSV access log."""
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Event', 'Person_Name', 'Details'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event, person_name, details
        ])

# ==============================================================================
#  REGISTRATION (Webcam & Bulk upload)
# ==============================================================================
def register_family_member(name: str, num_photos: int = 3):
    """
    Capture `num_photos` webcam shots for one family member.
    """
    name = name.strip().replace(" ", "_")

    person_folder = os.path.join(DATABASE_PATH, name)
    os.makedirs(person_folder, exist_ok=True)

    if name not in AUTHORIZED_FAMILY:
        AUTHORIZED_FAMILY[name] = []

    print(f"\n[USER] Registering: {name} ({num_photos} photo(s) requested)")
    successful = 0

    for i in range(1, num_photos + 1):
        print(f"\n  [CAMERA] Photo {i}/{num_photos} - look at the camera...")
        img = take_photo(
            window_title=f"Register '{name}' - photo {i}/{num_photos}  |  SPACE=capture  ESC=skip"
        )

        if img is None:
            print(f"  [WARN] Photo {i} skipped.")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(person_folder, filename)
        cv2.imwrite(filepath, img)
        print(f"  [SAVE] Image saved -> {filepath}")

        vector = get_embedding(img)
        if vector is None:
            print("  [WARN] No face detected in this photo - embedding skipped.")
            continue

        npy_path = filepath.replace(".jpg", ".npy")
        np.save(npy_path, vector)
        print(f"  [MODEL] Embedding saved -> {npy_path}")

        AUTHORIZED_FAMILY[name].append(vector)
        successful += 1

        log_event("Registration", name, filename)

    total = len(AUTHORIZED_FAMILY[name])
    if successful:
        print(f"\n[OK] '{name}' now has {total} total embedding(s) in the database.")
    else:
        print(f"\n[ERROR] No usable photos captured for '{name}'.")

def register_from_files(source_folder="./SmartLock/upload_queue"):
    """Detects faces in uploaded photos & adds them to the database without needing a live webcam session."""
    if not os.path.exists(source_folder):
        os.makedirs(source_folder, exist_ok=True)
        print(f"[INFO] Upload folder created at {source_folder}. Please add subfolders with names.")
        return

    print(f"\n[IMPORT] Processing files from {source_folder}...")

    for name in os.listdir(source_folder):
        person_source_path = os.path.join(source_folder, name)

        if not os.path.isdir(person_source_path):
            continue

        db_person_path = os.path.join(DATABASE_PATH, name)
        os.makedirs(db_person_path, exist_ok=True)

        for filename in os.listdir(person_source_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_source_path, filename)
                img = cv2.imread(img_path)

                if img is None:
                    os.remove(img_path)  # Remove unreadable files too
                    continue

                vector = get_embedding(img)

                if vector is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_img_name = f"{name}_upload_{timestamp}.jpg"
                    cv2.imwrite(os.path.join(db_person_path, new_img_name), img)

                    npy_name = new_img_name.replace(".jpg", ".npy")
                    np.save(os.path.join(db_person_path, npy_name), vector)

                    os.remove(img_path)  #Delete after successful processing
                    print(f"  [OK] Registered {filename} for {name}")
                else:
                    os.remove(img_path)  #Delete unusable photos too
                    print(f"  [SKIP] No face found in {filename} - ensure the person is looking at the camera.")

        # Remove the person's subfolder if it's now empty
        if not os.listdir(person_source_path):
            os.rmdir(person_source_path)
            print(f"  [CLEAN] Removed empty folder for {name}")

    load_database()

def register_mixed_group_photos(source_folder="./SmartLock/mixed_uploads"):
    """Process group photos with multiple faces for registration."""
    if not os.path.exists(source_folder):
        os.makedirs(source_folder, exist_ok=True)
        print(f"[INFO] Created folder at {source_folder}. Drop your group photos there!")
        return

    print(f"\n[IMPORT] Scanning {source_folder} for group photos...")

    for filename in os.listdir(source_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            os.remove(img_path)
            continue

        print(f"\n{'=' * 60}")
        print(f"[PROCESSING FILE]: {filename}")
        print("=" * 60)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_faces = face_cascade.detectMultiScale(gray, 1.02, 12, minSize=(100, 100))

        faces = []
        for (x, y, w, h) in raw_faces:
            aspect_ratio = float(w) / h
            if 0.75 <= aspect_ratio <= 1.3:
                faces.append((x, y, w, h))
            else:
                print(f"  [HAAR FILTER] Ignored artifact at ({x},{y}) aspect ratio {aspect_ratio:.2f}")

        print(f"  --> Found {len(faces)} confirmed human face(s) after geometric filtering.")

        if not faces:
            os.remove(img_path)
            print(f"  [CLEAN] No faces found. Removed source file: {filename}")
            continue

        # ── PHASE 1: Identify all faces first ──────────────────────────
        face_results = []  # list of dicts: {crop, crop_resized, vector, matched_name, distance, is_new}

        for i, (x, y, w, h) in enumerate(faces):
            face_crop = img[y:y + h, x:x + w]
            face_crop_resized = cv2.resize(face_crop, (250, 250))

            facenet_input = cv2.resize(face_crop, (160, 160))
            face_rgb = cv2.cvtColor(facenet_input, cv2.COLOR_BGR2RGB)
            batch = np.expand_dims(face_rgb, axis=0)
            vector = embedder.embeddings(batch)

            if vector is None:
                print(f"  [ERROR] Could not extract embedding for face {i + 1}. Skipping.")
                continue

            best_match_name = None
            lowest_distance = 999.0

            for family_name, saved_embeddings in AUTHORIZED_FAMILY.items():
                min_dist = min(np.linalg.norm(vector - sv) for sv in saved_embeddings)
                if min_dist < lowest_distance:
                    lowest_distance = min_dist
                    best_match_name = family_name

            STATIC_PHOTO_THRESHOLD = 0.72
            is_new = lowest_distance > STATIC_PHOTO_THRESHOLD

            face_results.append({
                "index": i + 1,
                "crop": face_crop,
                "crop_resized": face_crop_resized,
                "vector": vector,
                "matched_name": best_match_name if not is_new else None,
                "distance": lowest_distance,
                "is_new": is_new,
            })

        if not face_results:
            os.remove(img_path)
            continue

        # ── PHASE 2: Show all faces to user and confirm ────────────────
        print(f"\n{'─' * 60}")
        print(f"  [REVIEW] Showing all {len(face_results)} detected face(s) for '{filename}'")
        print(f"{'─' * 60}")

        for result in face_results:
            window_title = f"Face {result['index']}/{len(face_results)}"
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_title, result["crop_resized"])

        cv2.waitKey(1)  # Flush all windows at once

        print(f"\n  Summary of identifications:")
        for result in face_results:
            if result["is_new"]:
                print(f"    Face {result['index']}: ??? UNKNOWN  (closest was '{result['matched_name']}', dist={result['distance']:.4f})")
            else:
                print(f"    Face {result['index']}: {result['matched_name'].replace('_', ' ').upper()}  (dist={result['distance']:.4f})")

        print(f"\n  Are all identifications correct? (y/n): ", end="")
        confirmation = input().strip().lower()
        cv2.destroyAllWindows()

        if confirmation != "y":
            # Let user correct each face individually
            print(f"\n  [CORRECTION MODE] Re-examine each face:")
            for result in face_results:
                cv2.namedWindow(f"Correct Face {result['index']}", cv2.WINDOW_AUTOSIZE)
                cv2.imshow(f"Correct Face {result['index']}", result["crop_resized"])
                cv2.waitKey(1)

                if result["is_new"]:
                    print(f"  Face {result['index']} — currently UNKNOWN.")
                else:
                    print(f"  Face {result['index']} — currently identified as '{result['matched_name'].replace('_', ' ')}'.")

                user_input = input(f"  Enter correct name (or press Enter to keep current): ").strip()
                cv2.destroyWindow(f"Correct Face {result['index']}")

                if user_input:
                    result["matched_name"] = user_input.replace(" ", "_")
                    result["is_new"] = result["matched_name"] not in AUTHORIZED_FAMILY

        # ── PHASE 3: Handle new faces — ask for names ──────────────────
        for result in face_results:
            if result["is_new"] and result["matched_name"] is None:
                cv2.namedWindow(f"New Person — Face {result['index']}", cv2.WINDOW_AUTOSIZE)
                cv2.imshow(f"New Person — Face {result['index']}", result["crop_resized"])
                cv2.waitKey(1)

                print(f"\n  [NEW PERSON] Face {result['index']} is not in the database.")
                user_input = input(f"  Enter name for this person (or press Enter to skip): ").strip()
                cv2.destroyWindow(f"New Person — Face {result['index']}")

                if not user_input:
                    print(f"  [SKIPPED] Face {result['index']} ignored.")
                    result["matched_name"] = None  # Mark to skip saving
                else:
                    result["matched_name"] = user_input.replace(" ", "_")
                    print(f"  [NEW FOLDER] Will create new entry for '{result['matched_name']}'.")

        # ── PHASE 4: Save all confirmed faces ──────────────────────────
        print(f"\n  [SAVING] Writing embeddings to database...")
        for result in face_results:
            if result["matched_name"] is None:
                continue

            person_name = result["matched_name"]
            db_person_path = os.path.join(DATABASE_PATH, person_name)
            os.makedirs(db_person_path, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_img_name = f"{person_name}_group_crop_{timestamp}_{result['index']}.jpg"

            cv2.imwrite(os.path.join(db_person_path, new_img_name), result["crop_resized"])
            np.save(os.path.join(db_person_path, new_img_name.replace(".jpg", ".npy")), result["vector"])
            print(f"  [STORED] {person_name} ← face {result['index']}")

        # ── PHASE 5: Delete source photo ───────────────────────────────
        os.remove(img_path)
        print(f"  [CLEAN] Removed source file: {filename}")

    cv2.destroyAllWindows()
    load_database()
    print("\n[PROCESS COMPLETE] Database re-indexed.")

# ==============================================================================
#  LOAD EXISTING DATABASE FROM DISK
# ==============================================================================
def load_database():
    """
    Walk face_database/ and reload every .npy embedding saved by previous
    registration sessions.
    """
    AUTHORIZED_FAMILY.clear()

    if not os.path.isdir(DATABASE_PATH):
        print("[WARN] No database folder found - please register family members first.")
        return

    print("[DIR] Loading face database from disk...")
    for person_name in sorted(os.listdir(DATABASE_PATH)):
        person_folder = os.path.join(DATABASE_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue

        embeddings = []
        for fname in sorted(os.listdir(person_folder)):
            if fname.endswith(".npy"):
                vec = np.load(os.path.join(person_folder, fname))
                embeddings.append(vec)

        if embeddings:
            AUTHORIZED_FAMILY[person_name] = embeddings
            print(f"    [OK] '{person_name}' - {len(embeddings)} embedding(s) loaded")

    if AUTHORIZED_FAMILY:
        print(f"\n[OK] Database ready - {len(AUTHORIZED_FAMILY)} family member(s).\n")
    else:
        print("[WARN] Database is empty - please register family members first.\n")

# ==============================================================================
#  CLASSIFICATION
# ==============================================================================
def identify_person(live_vector, threshold=GLOBAL_THRESHOLD, k=3):
    """Compare live_vector against every stored embedding using KNN majority voting."""
    distances = []

    for name, vectors in AUTHORIZED_FAMILY.items():
        for stored_vec in vectors:
            dist = float(np.linalg.norm(live_vector - stored_vec))
            distances.append((dist, name))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    neighbor_names = [name for _, name in k_nearest]

    votes = Counter(neighbor_names)
    best_name = votes.most_common(1)[0][0]

    winning_distances = [dist for dist, name in k_nearest if name == best_name]
    avg_distance = sum(winning_distances) / len(winning_distances)

    if avg_distance >= threshold:
        return "STRANGER", avg_distance

    return best_name, avg_distance

# ==============================================================================
#  UNLOCK ATTEMPT (with liveness checks)
# ==============================================================================
def attempt_unlock(threshold=GLOBAL_THRESHOLD):
    """
    Capture a live frame, identify who is at the door, perform liveness checks,
    and decide UNLOCK / LOCK accordingly.
    """
    print("\n[LOCK] SMART LOCK ENGAGED - awaiting visitor...")

    if not AUTHORIZED_FAMILY:
        print("[WARN] No family members in database. Please register first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("  [Camera] SPACE = scan face  |  ESC = cancel")
    live_img = None
    live_vector = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  [Camera] Failed to grab frame.")
            break

        cv2.imshow("Smart Lock - SPACE to scan", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("  [Camera] Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == 32:
            live_vector = get_embedding(frame)
            if live_vector is None:
                print("  [ALERT] No face detected. Please try again.")
                continue
            print("  [Camera] [CAPTURE] Captured!")
            live_img = frame.copy()
            break

    cv2.destroyWindow("Smart Lock - SPACE to scan")

    name, distance = identify_person(live_vector, threshold, k=3)
    print(f"[STATS] Best match : '{name}'")
    print(f"[DIST] Distance   : {distance:.4f} (threshold = {threshold})")

    if name != "STRANGER":
        print(f"\n[LIVENESS] Face matched '{name}'. Initiating randomized liveness check...")

        # --- RANDOMIZED INTERACTIVE CHALLENGE ---
        # We pick a random number: 0 for Blink, 1 for Head Turn
        random_int = random.randint(0, 1)

        # 0 = Blink, 1 = Turn Left, 2 = Turn Right
        challenge_type = random.randint(0, 2)

        if challenge_type == 0:
            is_live = check_liveness_blink(cap=cap)
        elif challenge_type == 1:
            is_live = check_liveness_head_turn(cap=cap, target_side="LEFT")
        else:
            is_live = check_liveness_head_turn(cap=cap, target_side="RIGHT")

        # Check anomalies
        check_for_anomalies(name, distance)

        if not is_live:
            print("\n[ALERT] LIVENESS FAILED - System locked.")
            print("[SIGNAL] SENDING SIGNAL TO MICROPROCESSOR: [SERIAL: b'LOCK']")
            log_event("Liveness Failed", name, f"distance={distance:.4f} - possible spoof")
        else:
            display_name = name.replace("_", " ")
            print(f"\n[INFO] ACCESS GRANTED - Welcome home, {display_name}!")
            print("[SIGNAL] SENDING SIGNAL TO MICROPROCESSOR: [SERIAL: b'UNLOCK']")
            log_event("Access Granted", name, f"distance={distance:.4f}")
    else:
        print("\n[ALERT] STRANGER DANGER - Face not recognised.")
        print("[SIGNAL] SENDING SIGNAL TO MICROPROCESSOR: [SERIAL: b'LOCK']")
        log_event("Access Denied", "STRANGER", f"distance={distance:.4f}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        intruder_path = os.path.join(os.path.dirname(LOG_PATH), f"intruder_{ts}.jpg")
        cv2.imwrite(intruder_path, live_img)
        print(f"[CAMERA] Intruder snapshot saved -> {intruder_path}")
        send_alert(
            subject="[Smart Lock] Unrecognised person at the door",
            body=(
                f"Smart Lock Alert\n"
                f"─────────────────\n"
                f"An unrecognised visitor was detected.\n\n"
                f"Time : {datetime.now().strftime('%I:%M:%S %p')}\n"
                f"See attached photo.\n"
            ),
            image_path=intruder_path  # ← image is now attached, not just a path
        )

    cap.release()
    cv2.destroyAllWindows()

# ==============================================================================
#  EVALUATION FUNCTIONS
# ==============================================================================
def evaluate_knn(test_folder, threshold=0.6, k=3):
    """Evaluate KNN accuracy on test dataset."""
    total = 0
    correct = 0

    for person_name in os.listdir(test_folder):
        person_path = os.path.join(test_folder, person_name)

        if not os.path.isdir(person_path):
            continue

        for file in os.listdir(person_path):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)
            vector = get_embedding(img)

            if vector is None:
                continue

            predicted, distance = identify_person(vector, threshold=threshold, k=k)
            total += 1

            if predicted == person_name:
                correct += 1

            print(f"Actual: {person_name} | Predicted: {predicted} | Distance: {distance:.3f}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nK={k}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# ==============================================================================
#  LIVE CONTINUOUS RECOGNITION
# ==============================================================================
def auto_learn(name: str, frame: np.ndarray, vector: np.ndarray, distance: float):
    """
    Save a new photo + embedding for an already-recognised family member.
    """
    internal_name = name.replace(" ", "_")
    person_folder = os.path.join(DATABASE_PATH, internal_name)
    os.makedirs(person_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{internal_name}_autolearn_{timestamp}.jpg"
    filepath = os.path.join(person_folder, filename)

    cv2.imwrite(filepath, frame)
    npy_path = filepath.replace(".jpg", ".npy")
    np.save(npy_path, vector)

    AUTHORIZED_FAMILY[internal_name].append(vector)
    total = len(AUTHORIZED_FAMILY[internal_name])

    print(f"[LEARN] [{_timestamp()}] Auto-learned new look for {name} "
          f"-> {filename} (error/distance: {distance:.4f}, total embeddings: {total})")
    log_event("Auto-Learn", internal_name, f"file={filename}, dist={distance:.4f}")

def run_live_recognition(
        threshold=GLOBAL_THRESHOLD,
        learn_zone=0.35,
        cooldown_seconds=1.0,
        learn_cooldown=5.0,
):
    """
    Keep the camera open, greet whoever appears, and automatically learn
    new looks for recognised family members.
    """
    if not AUTHORIZED_FAMILY:
        print("[WARN] No family members in database. Please register first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("\n[LIVE] Live recognition started - press Q or ESC to quit\n")

    last_identification_time = 0.0
    current_label = ""
    last_printed_label = ""
    box_color = (128, 128, 128)
    last_learn_time: dict = {}
    stranger_strike_count = 0
    STRANGER_STRIKES_NEEDED = 3

    import time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost camera feed.")
            break

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        face_detected = len(faces) > 0

        if face_detected and (now - last_identification_time) >= cooldown_seconds:
            last_identification_time = now

            vector = get_embedding(frame)
            if vector is not None:
                name, distance = identify_person(vector, threshold, k=3)
                check_for_anomalies(name, distance)

                if name != "STRANGER":
                    stranger_strike_count = 0
                    display_name = name.replace("_", " ")
                    current_label = display_name
                    box_color = (0, 200, 0)

                    in_learn_zone = learn_zone <= distance < threshold
                    learn_cd_elapsed = (now - last_learn_time.get(name, 0)) >= learn_cooldown

                    if in_learn_zone and learn_cd_elapsed:
                        auto_learn(display_name, frame, vector, distance)
                        last_learn_time[name] = now
                        box_color = (0, 215, 255)
                else:
                    stranger_strike_count += 1
                    if stranger_strike_count >= STRANGER_STRIKES_NEEDED:
                        current_label = "Stranger"
                        box_color = (0, 0, 220)

                if current_label != last_printed_label:
                    if current_label == "Stranger":
                        print(f"[ALERT] [{_timestamp()}] Stranger detected!")
                    else:
                        print(f"[INFO] [{_timestamp()}] Hello, {current_label}! "
                              f"(distance: {distance:.3f})")
                    last_printed_label = current_label

        if not face_detected:
            if last_printed_label != "":
                print(f"[INFO] [{_timestamp()}] {last_printed_label} left the frame.")
            current_label = ""
            last_printed_label = ""
            box_color = (128, 128, 128)
            stranger_strike_count = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            if current_label:
                label_text = f"  {current_label}  "
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x, y - th - 12), (x + tw, y), box_color, -1)
                cv2.putText(frame, label_text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        h_frame = frame.shape[0]
        cv2.putText(frame, "Q / ESC = quit", (10, h_frame - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Smart Lock - Live Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            print("\n[INFO] Live recognition stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()

def _timestamp():
    """Helper: current time as 12-hour HH:MM:SS string."""
    return datetime.now().strftime("%I:%M:%S %p")

# ==============================================================================
#  ALERT CONFIGURATION — fill these in once
# ==============================================================================
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

ALERT_EMAIL_ENABLED  = True

GMAIL_SENDER         = "mariamrageh2@gmail.com"
GMAIL_APP_PASSWORD   = "vqei hopl xbfg jjqm"   # 16-char App Password from Google

OWNER_EMAIL          = "mariamfikry06@gmail.com"

def send_alert(subject: str, body: str, image_path: str = None):
    """Send an email and/or SMS to the house owner when an alarm triggers."""
    recipients = []
    if ALERT_EMAIL_ENABLED:
        recipients.append(OWNER_EMAIL)

    if not recipients:
        return

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)

            for recipient in recipients:
                msg = MIMEMultipart()
                msg["From"]    = GMAIL_SENDER
                msg["To"]      = recipient
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))

                # ── Attach image if provided and file exists ───────────
                if image_path and os.path.isfile(image_path):
                    with open(image_path, "rb") as img_file:
                        img_data = img_file.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    msg.attach(image)

                server.sendmail(GMAIL_SENDER, recipient, msg.as_string())

        print(f"[ALERT] Notification sent to {len(recipients)} recipient(s).")

    except Exception as e:
        print(f"[ALERT] Failed to send notification: {e}")
# ==============================================================================
#  MAIN - Interactive menu
# ==============================================================================
if __name__ == "__main__":

    # STEP 1: Load embeddings saved from previous sessions
    load_database()

    # STEP 2: Force registration if database is completely empty
    while not AUTHORIZED_FAMILY:
        print("\n[WARN] The family dataset is empty. Please register at least one person first.")
        print("\n  1) Register via Webcam (live photos)")
        print("  2) Bulk Upload from Folder (organised name folders)")
        print("  3) Mixed Group Photo Upload (interactive labelling)")
        print("  4) Exit")

        choice = input("\nChoice (1/2/3/4): ").strip()

        if choice == '1':
            new_name = input("\nEnter the name of the new family member: ").strip()
            if new_name:
                register_family_member(new_name, num_photos=3)
            else:
                print("[WARN] Name cannot be empty.")
        elif choice == '2':
            register_from_files()
        elif choice == '3':
            register_mixed_group_photos()
        elif choice == '4':
            print("Exiting...")
            sys.exit(0)
        else:
            print("[WARN] Invalid choice.")

    # Optional: KNN evaluation if test folder exists
    """
    if os.path.exists("./SmartLock/test_faces"):
        print("\n" + "=" * 50)
        print("  DIAGNOSTIC: KNN EVALUATION")
        print("=" * 50)
        for k in [1, 3, 5, 7]:
            evaluate_knn(test_folder="./SmartLock/test_faces", threshold=GLOBAL_THRESHOLD, k=k)

        print("\n" + "=" * 50)
        print("  DIAGNOSTIC: THRESHOLD SENSITIVITY SWEEP")
        print("=" * 50)
        for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            print(f"\n[TESTING] Threshold: {t}")
            evaluate_knn(test_folder="./SmartLock/test_faces", threshold=t, k=3)
        print("\n" + "=" * 50)
        print("  DIAGNOSTIC COMPLETE")
        print("=" * 50 + "\n")
    """
    # STEP 3: Main menu
    while True:
        member_names = [n.replace("_", " ") for n in sorted(AUTHORIZED_FAMILY.keys())]

        print("\n" + "=" * 50)
        print("  SMART LOCK — Main Menu")
        print("=" * 50)
        print(f"  Registered members ({len(member_names)}): {', '.join(member_names)}")
        print()
        print("  1) Registration mode")
        print("  2) Recognition mode")
        print("  3) Exit")
        print()

        mode = input("Choice (1/2/3): ").strip()

        # ── REGISTRATION MODE ──────────────────────────────────────────
        if mode == '1':
            while True:
                member_names = [n.replace("_", " ") for n in sorted(AUTHORIZED_FAMILY.keys())]

                print("\n" + "-" * 50)
                print("  Registration Mode")
                print("-" * 50)
                print(f"  Registered members ({len(member_names)}): {', '.join(member_names)}")
                print()
                print("  1) Register new member via Webcam (live photos)")
                print("  2) Bulk Upload from Folder (organised name folders)")
                print("  3) Mixed Group Photo Upload (interactive labelling)")
                print("  4) Back to Main Menu")
                print()

                reg_choice = input("Choice (1/2/3/4): ").strip()

                if reg_choice == '1':
                    new_name = input("\nEnter the name of the family member to register: ").strip()
                    if new_name:
                        register_family_member(new_name, num_photos=3)
                    else:
                        print("[WARN] Name cannot be empty.")

                elif reg_choice == '2':
                    register_from_files()

                elif reg_choice == '3':
                    register_mixed_group_photos()

                elif reg_choice == '4':
                    break

                else:
                    print("[WARN] Invalid choice.")

        # ── RECOGNITION MODE ───────────────────────────────────────────
        elif mode == '2':
            while True:
                print("\n" + "-" * 50)
                print("  Recognition Mode")
                print("-" * 50)
                print()
                print("  1) Live recognition (continuous)")
                print("  2) Single unlock attempt (with liveness checks)")
                print("  3) Back to Main Menu")
                print()

                rec_choice = input("Choice (1/2/3): ").strip()

                if rec_choice == '1':
                    run_live_recognition(
                        threshold=GLOBAL_THRESHOLD,
                        learn_zone=0.35,
                        cooldown_seconds=1.0,
                        learn_cooldown=5.0,
                    )

                elif rec_choice == '2':
                    while True:
                        attempt_unlock(threshold=GLOBAL_THRESHOLD)
                        again = input("\nPress ENTER to try again, or 'q' to return: ").strip().lower()
                        if again == 'q':
                            break

                elif rec_choice == '3':
                    break

                else:
                    print("[WARN] Invalid choice.")

        elif mode == '3':
            print("\nGoodbye!")
            sys.exit(0)

        else:
            print("[WARN] Invalid choice. Please enter 1, 2, or 3.")