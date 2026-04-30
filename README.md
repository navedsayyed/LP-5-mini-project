# LP-5 Mini Project

This repository contains two projects for the LP-5 End Semester submission:

1. **Deep Learning Project** — Real-time Face Recognition with Age & Gender Detection
2. **HPC Project** — Huffman Encoding/Decoding with GPU Acceleration

---

## 📁 Project Structure

```
endsem/
├── DeepLearning_Project/
│   ├── dl.py                  # Main face recognition + age/gender script
│   ├── download_models.py     # Downloads age & gender model weights
│   ├── Run_Face_Recognition.bat
│   ├── dataset/               # Known face images (one per person)
│   │   ├── Naved.jpeg
│   │   ├── Sam.jpeg
│   │   └── sujal_clean.jpg
│   └── models/                # Age & Gender DNN model files
│       ├── age_deploy.prototxt
│       ├── gender_deploy.prototxt
│       ├── age_net.caffemodel      (downloaded via script)
│       └── gender_net.caffemodel   (downloaded via script)
│
├── HPC_Project/
│   └── hpc.py                 # Huffman encoding with CuPy/NumPy
│
├── .gitignore
└── README.md
```

---

## 🧠 1. Deep Learning Project — Face Recognition + Age & Gender

Real-time webcam-based system that:
- **Recognizes faces** from a known dataset using `face_recognition` (dlib)
- **Predicts gender** (Male / Female) and **age range** using pre-trained Caffe CNNs
- Runs face detection in a **background thread** for smooth, lag-free camera feed

### How to Run (after cloning)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/endsem.git
cd endsem

# 2. Go to the DL project
cd DeepLearning_Project

# 3. Create a virtual environment & install dependencies
python -m venv dl_env
.\dl_env\Scripts\activate

pip install opencv-python face_recognition numpy dlib

# 4. Download age & gender model weights (~87 MB total)
python download_models.py

# 5. Run the project
$env:PYTHONIOENCODING="utf-8"; python dl.py
```

> **Note:** To add your own face, place a clear photo in the `dataset/` folder.
> The filename (without extension) becomes the label, e.g. `John.jpg` → **JOHN**.

### Controls
- Press **`q`** to quit the webcam window.

### Tech Stack
| Component | Library |
|---|---|
| Face Detection & Recognition | `face_recognition` (dlib) |
| Age & Gender Prediction | OpenCV DNN (Caffe models) |
| Camera & Display | OpenCV |

---

## ⚡ 2. HPC Project — Huffman Encoding with GPU Acceleration

Implements **Huffman Encoding and Decoding** with optional GPU acceleration using CuPy.
Falls back to NumPy automatically if no CUDA GPU is available.

Features:
- Character frequency analysis (GPU-accelerated with CuPy)
- Huffman tree construction & binary encoding
- Decoding back to original text
- Visual Huffman tree plot using NetworkX + Matplotlib

### How to Run (after cloning)

```bash
# 1. Go to the HPC project
cd HPC_Project

# 2. Install dependencies
pip install numpy matplotlib networkx

# (Optional) For GPU acceleration:
pip install cupy-cuda12x

# 3. Run the project
python hpc.py
```

> Enter any text when prompted. The program will display the frequency table,
> Huffman codes, encoded binary string, decoded output, and a tree visualization.

---

## 📋 Requirements

- Python 3.10+
- Webcam (for the DL project)
- Windows OS (tested on Windows 10/11)

