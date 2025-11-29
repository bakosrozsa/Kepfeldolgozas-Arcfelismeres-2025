import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data import random_split

# ------------------------------------------------------------------------ config ------------------------------------------------------------------------ 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "..", "project_data", "train")
IMAGE_FOLDER = os.path.join(DATA_ROOT, "image_data")
BBOX_CSV = os.path.join(DATA_ROOT, "bbox_train.csv")
OUTPUT_DIR = "models"
MODEL_NAME = "face_count_bbox_cnn.pth"
IMG_SIZE = (300, 300)
MAX_FACES = 10
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------------------------------------------------------------------------------------------------------------------

class FaceDataset(Dataset):
    def __init__(self, img_folder, bbox_csv, max_faces=MAX_FACES, img_size=IMG_SIZE):
        self.img_folder = img_folder
        df = pd.read_csv(bbox_csv)
        grouped = df.groupby("Name")
        self.records = []
        for name, group in grouped:
            boxes = group[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32)
            w = group.iloc[0]["width"]
            h = group.iloc[0]["height"]
            norm_boxes = boxes.copy()
            norm_boxes[:, [0,2]] /= w
            norm_boxes[:, [1,3]] /= h
            self.records.append( (name, norm_boxes) )
        self.max_faces = max_faces
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        name, boxes = self.records[idx]
        path = os.path.join(self.img_folder, name)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image {path}")
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32)
        img -= np.array([104.0, 177.0, 123.0])
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32)

        count = boxes.shape[0]
        count_norm = float(count) / self.max_faces
        # todo padding
        padded = np.zeros((self.max_faces, 4), dtype=np.float32)
        num = min(count, self.max_faces)
        padded[:num, :] = boxes[:num, :]
        target = torch.from_numpy( np.concatenate( ([count_norm], padded.flatten()) ) ).float()
        return img, target

class FaceCNN(nn.Module):
    def __init__(self, max_faces=MAX_FACES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out = nn.Linear(64, 1 + 4 * max_faces)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

def train():
    ds = FaceDataset(IMAGE_FOLDER, BBOX_CSV, max_faces=MAX_FACES, img_size=IMG_SIZE)

    test_ratio = 0.2
    test_size = int(len(ds) * test_ratio)
    train_size = len(ds) - test_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FaceCNN(max_faces=MAX_FACES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for imgs, targets in tqdm(train_loader, desc=f"Training {epoch+1}/{EPOCHS}", leave=False):
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= train_size
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f}")

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc="Evaluating on test", leave=False):
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, targets)
            test_loss += loss.item() * imgs.size(0)

    test_loss /= test_size
    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test MSE Loss: {test_loss:.6f}")
    print("========================================\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    print("Saved model to", save_path)

train()