import cv2
import os
import pandas as pd

# --- CONFIG ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'project_data', 'train'))

IMAGE_DIR = os.path.join(BASE_DIR, 'image_data')
CSV_HEADCOUNT = os.path.join(BASE_DIR, 'train.csv')
CSV_BBOX = os.path.join(BASE_DIR, 'bbox_train.csv')

# --- LOAD CSVs ---
headcount_df = pd.read_csv(CSV_HEADCOUNT)
bbox_df = pd.read_csv(CSV_BBOX)

# --- LOAD FACE DETECTOR ---
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# --- STATISTICS ---
total_images = 0
headcount_matches = 0
face_boxes_correct = 0
total_faces = 0
total_bboxes = 0  # csak feldolgozott képek bbox-aiból

under_detected = []  # túl kevés arc
over_detected = []  # túl sok arc

# --- LIST IMAGES ---
image_files = os.listdir(IMAGE_DIR)

# --- PROCESS IMAGES ---
for idx, img_name in enumerate(image_files, start=1):
    # --- CHECK IF IMAGE IS IN HEADCOUNT CSV ---
    headcount_row = headcount_df[headcount_df['Name'] == img_name]
    if headcount_row.empty:
        print(f"[INFO] {img_name} not found in train.csv, skipping")
        continue  # Nem számítjuk bele az összes képek számába

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARNING] Could not read image: {img_name}")
        continue

    total_images += 1
    true_headcount = int(headcount_row['HeadCount'].values[0])

    # --- DETECT FACES ---
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    detected_faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            box = box.astype(int)
            detected_faces.append(box)

    total_faces += len(detected_faces)

    # --- STATISTICS ---
    if len(detected_faces) == true_headcount:
        headcount_matches += 1
    elif len(detected_faces) < true_headcount:
        under_detected.append(img_name)
    else:
        over_detected.append(img_name)

    # --- BBOX ACCURACY ---
    bbox_rows = bbox_df[bbox_df['Name'] == img_name]
    total_bboxes += len(bbox_rows)
    for _, row in bbox_rows.iterrows():
        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        match_found = False
        for box in detected_faces:
            bx_min, by_min, bx_max, by_max = box
            dx = min(bx_max, x_max) - max(bx_min, x_min)
            dy = min(by_max, y_max) - max(by_min, y_min)
            if dx > 0 and dy > 0:
                overlap_area = dx * dy
                bbox_area = (x_max - x_min) * (y_max - y_min)
                if overlap_area / bbox_area > 0.5:
                    match_found = True
                    break
        if match_found:
            face_boxes_correct += 1

    # --- PROGRESS ---
    print(f"Processing image {total_images}/{len(image_files)} ({img_name}) - Detected faces: {len(detected_faces)}")

# --- FINAL STATISTICS ---
print("\n=== FINAL STATISTICS ===")
print(f"Total images processed: {total_images}")
print(f"Headcount exact matches: {headcount_matches}/{total_images} ({headcount_matches / total_images * 100:.2f}%)")
if total_bboxes > 0:
    print(f"Face boxes correct: {face_boxes_correct}/{total_bboxes} ({face_boxes_correct / total_bboxes * 100:.2f}%)")
print(f"Under-detected images: {len(under_detected)}")
print(f"Over-detected images: {len(over_detected)}")
