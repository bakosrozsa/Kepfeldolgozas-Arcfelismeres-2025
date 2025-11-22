import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QTextEdit, QFileDialog, QHBoxLayout, QLineEdit, QFormLayout
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QProcess, Qt

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arcfelismerés")
        self.setMinimumSize(800, 700)

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # -- Fájl beállítások --
        self.script_path = QLineEdit()
        self.script_path.setPlaceholderText("Válaszd ki a scriptet...")
        script_browse_btn = QPushButton("Tallózás")
        script_browse_btn.clicked.connect(
            lambda: self.browse_file(self.script_path, "Script keresése", "Python Files (*.py)"))

        script_layout = QHBoxLayout()
        script_layout.addWidget(self.script_path)
        script_layout.addWidget(script_browse_btn)
        form_layout.addRow("Python Script:", script_layout)

        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Válaszd ki a bemeneti képet...")
        image_browse_btn = QPushButton("Tallózás")
        image_browse_btn.clicked.connect(
            lambda: self.browse_file(self.image_path, "Kép keresése", "Image Files (*.png *.jpg *.bmp)"))

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_path)
        image_layout.addWidget(image_browse_btn)
        form_layout.addRow("Bemeneti kép:", image_layout)

        main_layout.addLayout(form_layout)

        # -- Kép előnézetek (Bemenet és Kimenet) ---
        preview_layout = QHBoxLayout()

        # Bemeneti kép
        input_preview_widget = QWidget()
        input_layout = QVBoxLayout(input_preview_widget)
        input_layout.addWidget(QLabel("Bemenet:"))
        self.input_preview = self.create_image_label()
        input_layout.addWidget(self.input_preview)
        preview_layout.addWidget(input_preview_widget)

        # Kimeneti kép
        output_preview_widget = QWidget()
        output_layout = QVBoxLayout(output_preview_widget)
        output_layout.addWidget(QLabel("Kimenet:"))
        self.output_preview = self.create_image_label()
        output_layout.addWidget(self.output_preview)
        preview_layout.addWidget(output_preview_widget)

        main_layout.addLayout(preview_layout)

        # -- Futtatás és Log --
        self.run_btn = QPushButton("Feldolgozás Indítása")
        self.run_btn.setFixedHeight(50)
        self.run_btn.setStyleSheet(
            "font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.run_btn.clicked.connect(self.start_script)
        self.run_btn.setEnabled(False)
        main_layout.addWidget(self.run_btn)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #222; color: #0f0; font-family: Consolas;")
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_output)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

    def create_image_label(self):
        """Segédfüggvény egy kép-label létrehozásához."""
        label = QLabel("Kép betöltése...")
        label.setMinimumHeight(250)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("border: 1px dashed #777; background-color: #f0f0f0;")
        return label

    def browse_file(self, target_input, window_title, file_filter):
        fname, _ = QFileDialog.getOpenFileName(self, window_title, "", file_filter)
        if fname:
            target_input.setText(fname)
            self.check_inputs()

            if target_input == self.image_path:
                self.load_image_to_label(fname, self.input_preview)
                self.output_preview.clear()
                self.output_preview.setText("Kimeneti kép...")
                self.output_preview.setStyleSheet("border: 1px dashed #777; background-color: #f0f0f0;")

    def load_image_to_label(self, image_path, target_label):
        """képbetöltő egy QLabel-be."""
        try:
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                target_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            target_label.setPixmap(scaled_pixmap)
            target_label.setStyleSheet("border: 1px solid #555;")
        except Exception as e:
            target_label.setText(f"Képbetöltési hiba: {e}")

    def check_inputs(self):
        script_ok = bool(self.script_path.text())
        image_ok = bool(self.image_path.text())
        self.run_btn.setEnabled(script_ok and image_ok)

    def start_script(self):
        self.log_output.clear()
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Feldolgozás folyamatban...")

        script = self.script_path.text()
        image = self.image_path.text()

        # A kimeneti fájl az 'output' mappába kerül
        self.output_file_path = os.path.join(OUTPUT_DIR, "result.jpg")

        self.log_output.append(f"Script indítása: {script}")
        self.log_output.append(f"Bemenet: {image}")
        self.log_output.append(f"Kimenet ide: {self.output_file_path}")
        self.log_output.append("-" * 30)

        # A parancs összerakása
        program = sys.executable
        args = [
            script,
            "--input", image,
            "--output", self.output_file_path
        ]

        self.process.start(program, args)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf8", errors="ignore")
        self.log_output.insertPlainText(data)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode("utf8", errors="ignore")
        self.log_output.append(f"<span style='color:red'>{data.strip()}</span>")

    def process_finished(self):
        self.log_output.append("\n<b>Folyamat befejeződött!</b>")
        self.run_btn.setText("Feldolgozás Indítása")
        self.check_inputs()

        # Betöltjük az elkészült képet a kimeneti ablakba
        self.log_output.append(f"Eredmény betöltése...")
        self.load_image_to_label(self.output_file_path, self.output_preview)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())