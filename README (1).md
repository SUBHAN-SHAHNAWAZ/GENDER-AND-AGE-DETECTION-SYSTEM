# 🎯 Gender and Age Detection System

A real-time gender and age detection system built from scratch using a custom CNN (Convolutional Neural Network) trained on the UTKFace dataset. Detects faces via webcam and predicts gender and age group in real time with bounding boxes.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Demo

> Real-time detection with green bounding box and label showing gender + age group on each detected face. Supports multiple faces in frame simultaneously.

---

## 🚀 Features

- ✅ Real-time face detection using OpenCV Haar Cascade
- ✅ Gender classification — Male / Female
- ✅ Age group classification — Child, Teen, Young Adult, Middle Age, Senior
- ✅ Supports multiple faces in one frame
- ✅ Custom CNN built from scratch — no pretrained weights
- ✅ Trained on 20,000+ real-world face images (UTKFace dataset)
- ✅ Works on CPU and GPU

---

## 🧠 Model Architecture

```
Input (3 × 128 × 128)
    │
    ├── Conv(3→32)   + ReLU + MaxPool  →  (32, 64, 64)
    ├── Conv(32→64)  + ReLU + MaxPool  →  (64, 32, 32)
    ├── Conv(64→128) + ReLU + MaxPool  →  (128, 16, 16)
    ├── Conv(128→256)+ ReLU + MaxPool  →  (256, 8,  8)
    │
    ├── AdaptiveAvgPool → (256, 4, 4)
    ├── Flatten → 4096
    │
    ├── Gender Head → Linear(4096→2)   →  Male / Female
    └── Age Head    → Linear(4096→5)   →  5 age groups
```

**Age Groups:**
| Label | Range |
|-------|-------|
| Child | 0 – 12 |
| Teen | 13 – 19 |
| Young Adult | 20 – 35 |
| Middle Age | 36 – 55 |
| Senior | 56+ |

---

## 📁 Project Structure

```
Gender And Age Detection System/
│
├── detect.py          ← Real-time webcam detection (run this)
├── best_model.pth     ← Trained model weights
├── requirements.txt   ← Dependencies
└── README.md
```

---

## ⚙️ Installation

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/your-username/gender-age-detection.git
cd gender-age-detection
```

**Step 2 — Install Python 3.11** from [python.org](https://www.python.org/downloads/release/python-3119/)

**Step 3 — Install dependencies:**
```bash
py -3.11 -m pip install torch torchvision opencv-python pillow
```

---

## ▶️ Run

```bash
py -3.11 detect.py
```

- A **webcam window** will open
- Every face in frame gets a **green bounding box**
- Label shows **Gender + Age Group** on top of each box
- Press **Q** to quit

---

## 🗃️ Dataset

Trained on the **UTKFace** dataset — 20,000+ face images labeled with age, gender and ethnicity.

- 📦 [Download from Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- Labels are encoded directly in filenames: `[age]_[gender]_[race]_[timestamp].jpg`

---

## 🏋️ Training (optional — model already provided)

If you want to retrain from scratch, open Google Colab and run the training notebook with the dataset path set to `/content/data/UTKFace`.

**Training details:**
| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=1e-3) |
| Loss | CrossEntropyLoss |
| Epochs | 20 |
| Batch size | 64 |
| Image size | 128×128 |
| Train/Val split | 85% / 15% |

**Expected results:**
| Metric | Accuracy |
|--------|----------|
| Gender | ~88–93% |
| Age Group | ~60–70% |

---

## 🛠️ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.0.0
pillow>=9.0.0
numpy>=1.23.0
```

---

## 📌 Notes

- `best_model.pth` must be in the **same folder** as `detect.py`
- Works best in **good lighting** with a clear front-facing view
- Age prediction is approximate by nature — even state-of-the-art models have a margin of error of a few years

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)

---

## 📄 License

This project is licensed under the MIT License.
