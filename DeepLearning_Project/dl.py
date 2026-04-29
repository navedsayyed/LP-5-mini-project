import cv2
import face_recognition
import os
import numpy as np
import sys
import threading

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_image(img):
    if img is None:
        return None

    img = np.asarray(img)

    # fix dtype
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 3D image
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return None
    else:
        return None

    # final safety check
    if img.ndim != 3 or img.shape[2] != 3:
        return None

    # dlib requires a C-contiguous array; cvtColor can produce non-contiguous views
    img = np.ascontiguousarray(img)

    return img


# ---------------- LOAD AGE & GENDER MODELS ----------------
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

age_net = None
gender_net = None

age_proto = 'models/age_deploy.prototxt'
age_model = 'models/age_net.caffemodel'
gender_proto = 'models/gender_deploy.prototxt'
gender_model = 'models/gender_net.caffemodel'

if os.path.exists(age_model) and os.path.exists(gender_model):
    age_net = cv2.dnn.readNet(age_proto, age_model)
    gender_net = cv2.dnn.readNet(gender_proto, gender_model)
    print("✅ Age & Gender models loaded")
else:
    print("⚠️  Age/Gender model weights not found.")
    print("   Run:  python download_models.py")
    print("   Continuing with face recognition only...\n")


# ---------------- LOAD DATASET ----------------
path = 'dataset'
images = []
classNames = []

for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):

        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)

        img = preprocess_image(img)

        if img is None:
            print(f"❌ Skipped (bad image): {file}")
            continue

        images.append(img)
        classNames.append(os.path.splitext(file)[0])
        print(f"✅ Loaded: {file}")


# ---------------- ENCODINGS ----------------
def findEncodings(images):
    encodeList = []

    for i, img in enumerate(images):
        try:
            print(f"Encoding: {classNames[i]} | Shape: {img.shape} | Dtype: {img.dtype}")

            encodes = face_recognition.face_encodings(img)

            if len(encodes) > 0:
                encodeList.append(encodes[0])
            else:
                print(f"⚠️ No face found in: {classNames[i]}")

        except Exception as e:
            print(f"❌ Encoding error ({classNames[i]}): {e}")

    return encodeList


encodeListKnown = findEncodings(images)
print("\n✅ Encoding Complete\n")


# ---------------- PREDICT AGE & GENDER ----------------
def predict_age_gender(face_bgr):
    """Takes a BGR face crop, returns (gender_str, age_str)."""
    if age_net is None or gender_net is None:
        return ("", "")

    blob = cv2.dnn.blobFromImage(
        face_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
    )

    # Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]

    return (gender, age)


# ---------------- WEBCAM ----------------
cam_index = 0
if len(sys.argv) > 1:
    try:
        cam_index = int(sys.argv[1])
    except ValueError:
        pass

print(f"Opening camera index: {cam_index}...")
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ----------- THREADED FACE PROCESSING -----------
lock = threading.Lock()
latest_frame = None
display_results = []       # list of (name, gender, age, faceLoc)
processing_busy = False
stop_event = threading.Event()


def face_processing_thread():
    """Detects faces, recognises identity, and predicts age+gender
    in a background thread so the camera feed stays smooth."""
    global latest_frame, display_results, processing_busy

    while not stop_event.is_set():
        with lock:
            frame = latest_frame
            latest_frame = None

        if frame is None:
            stop_event.wait(0.01)
            continue

        processing_busy = True

        # --- face detection on a small frame ---
        small = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        rgb_small = preprocess_image(small)

        if rgb_small is None:
            processing_busy = False
            continue

        rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        results = []
        for encode_face, face_loc in zip(face_encodings, face_locations):
            # --- Name recognition ---
            name = "UNKNOWN"
            if len(encodeListKnown) > 0:
                matches = face_recognition.compare_faces(
                    encodeListKnown, encode_face, tolerance=0.5
                )
                face_dis = face_recognition.face_distance(encodeListKnown, encode_face)
                match_index = np.argmin(face_dis)
                if matches[match_index]:
                    name = classNames[match_index].upper()

            # --- Age & Gender prediction ---
            # Crop face from the ORIGINAL full-size frame (better quality)
            top, right, bottom, left = face_loc
            top    = max(0, top * 4 - 20)
            right  = min(frame.shape[1], right * 4 + 20)
            bottom = min(frame.shape[0], bottom * 4 + 20)
            left   = max(0, left * 4 - 20)

            face_crop = frame[top:bottom, left:right]
            gender, age = predict_age_gender(face_crop)

            results.append((name, gender, age, face_loc))

        with lock:
            display_results = results

        processing_busy = False


worker = threading.Thread(target=face_processing_thread, daemon=True)
worker.start()

# ----------- MAIN LOOP (camera + drawing only) -----------
frame_count = 0

while True:
    success, img = cap.read()

    if not success or img is None:
        continue

    frame_count += 1
    if frame_count % 3 == 0 and not processing_busy:
        with lock:
            latest_frame = img.copy()

    # Draw cached results
    with lock:
        results_snapshot = list(display_results)

    for name, gender, age, face_loc in results_snapshot:
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Green box around the face
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- TOP label: Gender (Age) ---
        if gender and age:
            top_label = f"{gender} {age}"
            cv2.rectangle(img, (x1, y1 - 35), (x2, y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, top_label, (x1 + 6, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        # --- BOTTOM label: Name ---
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)

    cv2.imshow('Face Recognition + Age & Gender', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_event.set()
worker.join(timeout=2)
cap.release()
cv2.destroyAllWindows()