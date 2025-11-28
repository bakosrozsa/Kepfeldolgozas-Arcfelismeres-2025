import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# bele kell nyúlni ha nem gui-ból van futtatva
MODEL_PATH = os.path.join(PROJECT_ROOT,"scripts", "models", "face_count_bbox_cnn.pth")

IMG_SIZE = (300, 300)
MAX_FACES = 10

class FaceCNN(nn.Module):
    def __init__(self, max_faces=MAX_FACES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out = nn.Linear(64, 1 + 4 * max_faces)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

def preprocess_image(img_path):
    print(f"[INFO] Loading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load input image: {img_path}")

    orig = img.copy()
    orig_h, orig_w = img.shape[:2]

    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = img_resized.astype(np.float32)
    img_resized -= np.array([104.0, 177.0, 123.0])
    img_resized = img_resized.transpose(2, 0, 1)

    tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0)
    return orig, (orig_w, orig_h), tensor

def draw_boxes(img, boxes, count):
    h, w = img.shape[:2]
    print(f"[INFO] Drawing {count} boxes...")
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input image path")
parser.add_argument("--output", required=True, help="Output image path")
args = parser.parse_args()

print(f"[INFO] Loading model from {MODEL_PATH}")
model = FaceCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

orig_img, (orig_w, orig_h), tensor = preprocess_image(args.input)

print("[INFO] Running inference...")
with torch.no_grad():
    pred = model(tensor)[0].cpu().numpy()

count_norm = pred[0]
predicted_count = int(round(count_norm * MAX_FACES))
print(f"[INFO] Predicted face count: {predicted_count}")

box_values = pred[1:]
boxes = box_values.reshape(MAX_FACES, 4)
boxes = boxes[:predicted_count]

output_img = draw_boxes(orig_img, boxes, predicted_count)

os.makedirs(os.path.dirname(args.output), exist_ok=True)
cv2.imwrite(args.output, output_img)
print(f"[INFO] Saved result to: {args.output}")
print("[DONE]")