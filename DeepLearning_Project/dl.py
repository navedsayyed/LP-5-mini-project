import cv2
import face_recognition
import os
import numpy as np

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


import sys
import threading

# ---------------- WEBCAM ----------------
# Use index 0 by default, or take it from command line argument
cam_index = 0
if len(sys.argv) > 1:
    try:
        cam_index = int(sys.argv[1])
    except ValueError:
        pass

print(f"Opening camera index: {cam_index}...")
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize internal buffer lag

# ----------- THREADED FACE PROCESSING -----------
# Shared state between the background thread and main loop
lock = threading.Lock()
latest_frame = None          # frame to process (set by main loop)
display_results = []         # list of (name, faceLoc) ready to draw
processing_busy = False      # True while the bg thread is working
stop_event = threading.Event()


def face_processing_thread():
    """Runs in background: picks up frames, detects + recognises faces,
    and writes results back into display_results without blocking the
    main camera loop."""
    global latest_frame, display_results, processing_busy

    while not stop_event.is_set():
        # grab a frame to process
        with lock:
            frame = latest_frame
            latest_frame = None  # consume it

        if frame is None:
            # nothing to do – sleep briefly so we don't spin-lock
            stop_event.wait(0.01)
            continue

        processing_busy = True

        # --- heavy work happens here (off the main thread) ---
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
            if len(encodeListKnown) == 0:
                results.append(("UNKNOWN", face_loc))
                continue

            matches = face_recognition.compare_faces(encodeListKnown, encode_face, tolerance=0.5)
            face_dis = face_recognition.face_distance(encodeListKnown, encode_face)
            match_index = np.argmin(face_dis)

            name = "UNKNOWN"
            if matches[match_index]:
                name = classNames[match_index].upper()

            results.append((name, face_loc))

        # publish results atomically
        with lock:
            display_results = results

        processing_busy = False


# start the worker
worker = threading.Thread(target=face_processing_thread, daemon=True)
worker.start()

# ----------- MAIN LOOP (camera + drawing only) -----------
frame_count = 0

while True:
    success, img = cap.read()

    if not success or img is None:
        continue

    # Feed a frame to the background thread every 3 frames
    # (only if it is not still busy with the previous one)
    frame_count += 1
    if frame_count % 3 == 0 and not processing_busy:
        with lock:
            latest_frame = img.copy()

    # Draw the latest cached results – zero heavy computation here
    with lock:
        results_snapshot = list(display_results)

    for name, face_loc in results_snapshot:
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)

    cv2.imshow('Recognized', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_event.set()
worker.join(timeout=2)
cap.release()
cv2.destroyAllWindows()