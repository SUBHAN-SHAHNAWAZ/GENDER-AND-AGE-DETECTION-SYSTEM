import cv2
import torch
import torch.nn as nn
import numpy as np
import urllib.request
import torchvision.transforms as transforms
from PIL import Image

# ── Labels ─────────────────────────────────────────────────
AGE_GROUPS    = {0:"Child(0-12)", 1:"Teen(13-19)", 2:"Young Adult(20-35)", 3:"Middle Age(36-55)", 4:"Senior(56+)"}
GENDER_LABELS = {0:"Male", 1:"Female"}

# ── Transform ──────────────────────────────────────────────
tf_pred = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ── Model ──────────────────────────────────────────────────
class AgeGenderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,   32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,  64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.age_head    = nn.Linear(256*4*4, 5)
        self.gender_head = nn.Linear(256*4*4, 2)

    def forward(self, x):
        x = self.cnn(x).flatten(1)
        return self.age_head(x), self.gender_head(x)

# ── Load model ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AgeGenderCNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("✓ Model loaded!")

# ── Load face detector ─────────────────────────────────────
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print("✓ Face detector loaded!")

# ── Open webcam ────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("✓ Camera opened! Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    for (x, y, w, h) in faces:
        # Crop and predict
        face_crop = frame[y:y+h, x:x+w]
        img       = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        tensor    = tf_pred(img).unsqueeze(0).to(device)

        with torch.no_grad():
            age_out, gen_out = model(tensor)

        age_idx  = age_out.argmax(1).item()
        gen_idx  = gen_out.argmax(1).item()
        age_conf = torch.softmax(age_out, 1)[0][age_idx].item() * 100
        gen_conf = torch.softmax(gen_out, 1)[0][gen_idx].item() * 100
        label    = f"{GENDER_LABELS[gen_idx]}, {AGE_GROUPS[age_idx]}"

        # Green box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y-th-10), (x+tw+8, y), (0,255,0), -1)
        cv2.putText(frame, label, (x+4, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # Show live window
    cv2.imshow("Gender & Age Detection  —  Press Q to quit", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped!")