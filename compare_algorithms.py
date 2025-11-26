import cv2
import os
import pandas as pd
import numpy as np
import sys
import argparse
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
                             QTextEdit, QPushButton, QProgressBar, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import threading

# -- Modell beállítások --
PROTOTXT_PATH = "pretrained_models/deploy.prototxt"
MODEL_PATH = "pretrained_models/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5

# --- CONFIG ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_data', 'train'))
IMAGE_DIR = os.path.join(BASE_DIR, 'image_data')
CSV_HEADCOUNT = os.path.join(BASE_DIR, 'train.csv')
CSV_BBOX = os.path.join(BASE_DIR, 'bbox_train.csv')


def convert_to_grayscale(color_image):
    """
    Egy betöltött BGR (színes) képet alakít át szürkeárnyalatossá
    és visszaadja
    """
    return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


def detect_faces_haar(image):
    """
    Arcokat detektál Haar Cascade-dal egy képen.
    Visszatér: (arcok listája, feldolgozott kép)
    """

    # Modellek betöltése
    frontal_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'

    frontal_face_cascade = cv2.CascadeClassifier(frontal_cascade_path)
    profile_face_cascade = cv2.CascadeClassifier(profile_cascade_path)

    if frontal_face_cascade.empty() or profile_face_cascade.empty():
        print("Hiba: Haar Cascade modell(ek) betöltése sikertelen.")
        return [], image

    framed_image = image.copy()
    gray_image = convert_to_grayscale(image)

    faces_frontal = frontal_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_profile = profile_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    all_faces = list(faces_frontal) + list(faces_profile)

    for (x, y, w, h) in all_faces:
        cv2.rectangle(framed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return all_faces, framed_image


def detect_faces_dnn(image):
    """
    Arcokat detektál a DNN modellel egy képen.
    Visszatér: (arcok listája, feldolgozott kép)
    """

    # --Modell betöltése--
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    except cv2.error as e:
        print(f"Hiba: A DNN modell betöltése sikertelen.")
        print(f"Ellenőrizd az útvonalakat: {PROTOTXT_PATH} és {MODEL_PATH}")
        print(f"Részletek: {e}")
        return [], image

    framed_image = image.copy()
    (h, w) = image.shape[:2]

    # --Kép előkészítése--
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # --Detektálás futtatása--
    net.setInput(blob)
    detections = net.forward()

    detected_faces = []

    # --Eredmények feldolgozása--
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # --Szűrés a biztossági küszöb alapján--
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append((startX, startY, endX, endY))

            cv2.rectangle(framed_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return detected_faces, framed_image


def calculate_bbox_accuracy(detected_faces, bbox_rows):
    """
    Kiszámítja a bounding box pontosságot.
    """
    correct_boxes = 0
    total_bboxes = len(bbox_rows)

    for _, row in bbox_rows.iterrows():
        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        match_found = False
        for box in detected_faces:
            if isinstance(box, tuple) and len(box) == 4:  # DNN format: (startX, startY, endX, endY)
                bx_min, by_min, bx_max, by_max = box
            elif isinstance(box, np.ndarray) and len(box) == 4:  # Haar format: [x, y, w, h]
                bx_min, by_min, bx_max, by_max = box[0], box[1], box[0] + box[2], box[1] + box[3]
            else:
                continue

            dx = min(bx_max, x_max) - max(bx_min, x_min)
            dy = min(by_max, y_max) - max(by_min, y_min)
            if dx > 0 and dy > 0:
                overlap_area = dx * dy
                bbox_area = (x_max - x_min) * (y_max - y_min)
                if bbox_area > 0 and overlap_area / bbox_area > 0.5:
                    match_found = True
                    break
        if match_found:
            correct_boxes += 1

    return correct_boxes, total_bboxes


class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    processing_finished = pyqtSignal(dict, dict)
    stats_updated = pyqtSignal(dict, dict)  # Új signal a részleges eredményekhez

    def __init__(self):
        super().__init__()
        self.paused = False
        self.stopped = False
        self.pause_event = threading.Event()
        self.pause_event.set()  # Kezdetben nem paused

        self.stats_haar = {
            'total_images': 0,
            'headcount_matches': 0,
            'face_boxes_correct': 0,
            'total_faces': 0,
            'total_bboxes': 0,
            'under_detected': [],
            'over_detected': []
        }

        self.stats_dnn = {
            'total_images': 0,
            'headcount_matches': 0,
            'face_boxes_correct': 0,
            'total_faces': 0,
            'total_bboxes': 0,
            'under_detected': [],
            'over_detected': []
        }

    def run(self):
        # --- LOAD CSVs ---
        try:
            headcount_df = pd.read_csv(CSV_HEADCOUNT)
            bbox_df = pd.read_csv(CSV_BBOX)
        except FileNotFoundError as e:
            self.progress_updated.emit(0, f"Hiba: CSV fájlok nem találhatók - {e}")
            return

        # --- LIST IMAGES ---
        try:
            image_files = os.listdir(IMAGE_DIR)
        except FileNotFoundError:
            self.progress_updated.emit(0, f"Hiba: Kép könyvtár nem található - {IMAGE_DIR}")
            return

        total_files = len(image_files)

        # --- PROCESS IMAGES ---
        for idx, img_name in enumerate(image_files, start=1):
            # Szüneteltetés ellenőrzése
            self.pause_event.wait()

            if self.stopped:
                break

            # --- CHECK IF IMAGE IS IN HEADCOUNT CSV ---
            headcount_row = headcount_df[headcount_df['Name'] == img_name]
            if headcount_row.empty:
                self.progress_updated.emit(int(idx / total_files * 100), f"[INFO] {img_name} not found in train.csv, skipping")
                continue  # Nem számítjuk bele az összes képek számába

            img_path = os.path.join(IMAGE_DIR, img_name)
            image = cv2.imread(img_path)
            if image is None:
                self.progress_updated.emit(int(idx / total_files * 100), f"[WARNING] Could not read image: {img_name}")
                continue

            self.stats_haar['total_images'] += 1
            self.stats_dnn['total_images'] += 1
            true_headcount = int(headcount_row['HeadCount'].values[0])

            # --- DETECT FACES WITH HAAR ---
            faces_haar, _ = detect_faces_haar(image)
            self.stats_haar['total_faces'] += len(faces_haar)

            # --- DETECT FACES WITH DNN ---
            faces_dnn, _ = detect_faces_dnn(image)
            self.stats_dnn['total_faces'] += len(faces_dnn)

            # --- STATISTICS FOR HAAR ---
            if len(faces_haar) == true_headcount:
                self.stats_haar['headcount_matches'] += 1
            elif len(faces_haar) < true_headcount:
                self.stats_haar['under_detected'].append(img_name)
            else:
                self.stats_haar['over_detected'].append(img_name)

            # --- STATISTICS FOR DNN ---
            if len(faces_dnn) == true_headcount:
                self.stats_dnn['headcount_matches'] += 1
            elif len(faces_dnn) < true_headcount:
                self.stats_dnn['under_detected'].append(img_name)
            else:
                self.stats_dnn['over_detected'].append(img_name)

            # --- BBOX ACCURACY ---
            bbox_rows = bbox_df[bbox_df['Name'] == img_name]

            correct_haar, total_bbox_haar = calculate_bbox_accuracy(faces_haar, bbox_rows)
            self.stats_haar['face_boxes_correct'] += correct_haar
            self.stats_haar['total_bboxes'] += total_bbox_haar

            correct_dnn, total_bbox_dnn = calculate_bbox_accuracy(faces_dnn, bbox_rows)
            self.stats_dnn['face_boxes_correct'] += correct_dnn
            self.stats_dnn['total_bboxes'] += total_bbox_dnn

            # --- PROGRESS ---
            progress = int(idx / total_files * 100)
            self.progress_updated.emit(progress, f"Processing image {self.stats_haar['total_images']}/{len([f for f in image_files if f in headcount_df['Name'].values])} ({img_name}) - Haar: {len(faces_haar)}, DNN: {len(faces_dnn)}")

            # Részleges eredmények küldése minden kép után
            self.stats_updated.emit(self.stats_haar.copy(), self.stats_dnn.copy())

        self.processing_finished.emit(self.stats_haar, self.stats_dnn)

    def pause(self):
        """Szünetelteti a feldolgozást"""
        self.paused = True
        self.pause_event.clear()

    def resume(self):
        """Folytatja a feldolgozást"""
        self.paused = False
        self.pause_event.set()

    def stop(self):
        """Leállítja a feldolgozást"""
        self.stopped = True
        self.resume()  # Biztosítjuk hogy ne maradjon paused állapotban


class ComparisonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arcfelismerő Algoritmusok Összehasonlítása")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Progress bar and control buttons
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Feldolgozás Indítása")
        self.start_button.clicked.connect(self.start_processing)

        self.pause_button = QPushButton("Szüneteltetés")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Készen áll a feldolgozásra...")

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.progress_label)

        layout.addLayout(control_layout)

        # Create splitter for results
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Haar Cascade results
        haar_widget = QWidget()
        haar_layout = QVBoxLayout(haar_widget)
        haar_layout.addWidget(QLabel("Haar Cascade Eredmények"))
        self.haar_table = QTableWidget()
        self.haar_table.setColumnCount(2)
        self.haar_table.setHorizontalHeaderLabels(["Metrika", "Érték"])
        self.haar_table.horizontalHeader().setStretchLastSection(True)
        haar_layout.addWidget(self.haar_table)
        splitter.addWidget(haar_widget)

        # DNN results
        dnn_widget = QWidget()
        dnn_layout = QVBoxLayout(dnn_widget)
        dnn_layout.addWidget(QLabel("DNN Eredmények"))
        self.dnn_table = QTableWidget()
        self.dnn_table.setColumnCount(2)
        self.dnn_table.setHorizontalHeaderLabels(["Metrika", "Érték"])
        self.dnn_table.horizontalHeader().setStretchLastSection(True)
        dnn_layout.addWidget(self.dnn_table)
        splitter.addWidget(dnn_widget)

        layout.addWidget(splitter)

        # Comparison text area
        layout.addWidget(QLabel("Összehasonlítás és Elemzés"))
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        layout.addWidget(self.comparison_text)

        self.processing_thread = ProcessingThread()
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.display_results)
        self.processing_thread.stats_updated.connect(self.update_partial_results)

        self.is_processing = False
        self.is_paused = False

    def start_processing(self):
        if not self.is_processing:
            # Új feldolgozás indítása
            self.is_processing = True
            self.is_paused = False
            self.start_button.setText("Leállítás")
            self.pause_button.setEnabled(True)
            self.pause_button.setText("Szüneteltetés")
            self.progress_bar.setValue(0)
            self.processing_thread.start()
        else:
            # Feldolgozás leállítása
            self.processing_thread.stop()
            self.is_processing = False
            self.start_button.setText("Feldolgozás Indítása")
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.progress_label.setText("Feldolgozás leállítva")

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def toggle_pause(self):
        if self.is_paused:
            # Folytatás
            self.processing_thread.resume()
            self.is_paused = False
            self.pause_button.setText("Szüneteltetés")
            self.progress_label.setText("Feldolgozás folytatódik...")
        else:
            # Szüneteltetés
            self.processing_thread.pause()
            self.is_paused = True
            self.pause_button.setText("Folytatás")
            self.progress_label.setText("Feldolgozás szünetelve - eredmények frissítve")

    def update_partial_results(self, stats_haar, stats_dnn):
        """Részleges eredmények frissítése szüneteltetéskor"""
        if self.is_paused:
            # Display current results
            self.display_stats_in_table(self.haar_table, stats_haar, "Haar Cascade")
            self.display_stats_in_table(self.dnn_table, stats_dnn, "DNN")
            self.generate_comparison(stats_haar, stats_dnn)

    def display_results(self, stats_haar, stats_dnn):
        self.is_processing = False
        self.start_button.setText("Feldolgozás Indítása")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.progress_label.setText("Feldolgozás befejezve!")

        # Display Haar results
        self.display_stats_in_table(self.haar_table, stats_haar, "Haar Cascade")

        # Display DNN results
        self.display_stats_in_table(self.dnn_table, stats_dnn, "DNN")

        # Generate comparison
        self.generate_comparison(stats_haar, stats_dnn)

    def display_stats_in_table(self, table, stats, algorithm_name):
        table.setRowCount(0)
        rows = []

        total_images = stats['total_images']
        if total_images > 0:
            headcount_accuracy = stats['headcount_matches'] / total_images * 100
        else:
            headcount_accuracy = 0

        rows.append(["Összes feldolgozott kép", str(stats['total_images'])])
        rows.append(["Pontos fejlétszám egyezések", f"{stats['headcount_matches']}/{total_images} ({headcount_accuracy:.2f}%)"])

        if stats['total_bboxes'] > 0:
            bbox_accuracy = stats['face_boxes_correct'] / stats['total_bboxes'] * 100
            rows.append(["Helyes bounding boxok", f"{stats['face_boxes_correct']}/{stats['total_bboxes']} ({bbox_accuracy:.2f}%)"])
        else:
            rows.append(["Helyes bounding boxok", "N/A"])

        rows.append(["Összes detektált arc", str(stats['total_faces'])])
        rows.append(["Túl kevés arc detektálva", str(len(stats['under_detected']))])
        rows.append(["Túl sok arc detektálva", str(len(stats['over_detected']))])

        table.setRowCount(len(rows))
        for i, (metric, value) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(metric))
            table.setItem(i, 1, QTableWidgetItem(value))

        table.resizeColumnsToContents()

    def generate_comparison(self, stats_haar, stats_dnn):
        comparison_text = "<h2>Algoritmusok Összehasonlítása</h2>"

        total_images = stats_haar['total_images']

        if total_images > 0:
            haar_headcount_acc = stats_haar['headcount_matches'] / total_images * 100
            dnn_headcount_acc = stats_dnn['headcount_matches'] / total_images * 100

            comparison_text += f"<h3>Fejlétszám Pontosság</h3>"
            comparison_text += f"<p>Haar Cascade: {haar_headcount_acc:.2f}%</p>"
            comparison_text += f"<p>DNN: {dnn_headcount_acc:.2f}%</p>"

            if dnn_headcount_acc > haar_headcount_acc:
                comparison_text += f"<p><b>A DNN algoritmus pontosabb fejlétszám becslésben ({dnn_headcount_acc - haar_headcount_acc:.2f}% különbség).</b></p>"
            elif haar_headcount_acc > dnn_headcount_acc:
                comparison_text += f"<p><b>A Haar Cascade algoritmus pontosabb fejlétszám becslésben ({haar_headcount_acc - dnn_headcount_acc:.2f}% különbség).</b></p>"
            else:
                comparison_text += f"<p><b>A két algoritmus fejlétszám pontossága azonos.</b></p>"

        if stats_haar['total_bboxes'] > 0 and stats_dnn['total_bboxes'] > 0:
            haar_bbox_acc = stats_haar['face_boxes_correct'] / stats_haar['total_bboxes'] * 100
            dnn_bbox_acc = stats_dnn['face_boxes_correct'] / stats_dnn['total_bboxes'] * 100

            comparison_text += f"<h3>Bounding Box Pontosság</h3>"
            comparison_text += f"<p>Haar Cascade: {haar_bbox_acc:.2f}%</p>"
            comparison_text += f"<p>DNN: {dnn_bbox_acc:.2f}%</p>"

            if dnn_bbox_acc > haar_bbox_acc:
                comparison_text += f"<p><b>A DNN algoritmus pontosabb bounding box detektálásban ({dnn_bbox_acc - haar_bbox_acc:.2f}% különbség).</b></p>"
            elif haar_bbox_acc > dnn_bbox_acc:
                comparison_text += f"<p><b>A Haar Cascade algoritmus pontosabb bounding box detektálásban ({haar_bbox_acc - dnn_bbox_acc:.2f}% különbség).</b></p>"
            else:
                comparison_text += f"<p><b>A két algoritmus bounding box pontossága azonos.</b></p>"

        comparison_text += f"<h3>Detektált Arcok Száma</h3>"
        comparison_text += f"<p>Haar Cascade: {stats_haar['total_faces']} arc</p>"
        comparison_text += f"<p>DNN: {stats_dnn['total_faces']} arc</p>"

        comparison_text += f"<h3>Hibák</h3>"
        comparison_text += f"<p>Haar Cascade - Túl kevés: {len(stats_haar['under_detected'])}, Túl sok: {len(stats_haar['over_detected'])}</p>"
        comparison_text += f"<p>DNN - Túl kevés: {len(stats_dnn['under_detected'])}, Túl sok: {len(stats_dnn['over_detected'])}</p>"

        self.comparison_text.setHtml(comparison_text)


def main():
    app = QApplication(sys.argv)
    window = ComparisonWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
