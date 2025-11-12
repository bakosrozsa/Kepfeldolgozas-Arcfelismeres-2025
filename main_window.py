import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QTextEdit, QFileDialog, QHBoxLayout, QLineEdit, QFormLayout
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arcfelismerés")
        self.setMinimumSize(600, 700)

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # -- script választás --
        self.script_path = QLineEdit()
        self.script_path.setPlaceholderText("Válaszd ki a .py fájlt...")
        script_browse_btn = QPushButton("Tallózás")
        script_browse_btn.clicked.connect(
            lambda: self.browse_file(self.script_path, "Script keresése", "Python Files (*.py)"))

        script_layout = QHBoxLayout()
        script_layout.addWidget(self.script_path)
        script_layout.addWidget(script_browse_btn)
        form_layout.addRow("Python Script:", script_layout)

        # -- kép választás --
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

        # --- Kép előnézete ---
        main_layout.addWidget(QLabel("Kép előnézete:"))
        self.image_preview = QLabel("Itt fog megjelenni a kép")
        self.image_preview.setMinimumHeight(250)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet("""
            QLabel {
                border: 1px dashed #777;
                background-color: #f0f0f0;
                color: #888;
            }
        """)
        main_layout.addWidget(self.image_preview)

        # -- Futtatás gomb --
        self.run_btn = QPushButton("Script Indítása")
        self.run_btn.setFixedHeight(50)
        self.run_btn.setStyleSheet(
            "font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.run_btn.clicked.connect(self.start_script)
        main_layout.addWidget(self.run_btn)

        # -- Log --
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #222; color: #0f0; font-family: Consolas;")
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_output)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def browse_file(self, target_input, window_title, file_filter):
        fname, _ = QFileDialog.getOpenFileName(self, window_title, "", file_filter)
        if fname:
            target_input.setText(fname)

            # --- Kép előnézet frissítése ---
            if target_input == self.image_path:
                try:
                    pixmap = QPixmap(fname)

                    scaled_pixmap = pixmap.scaled(
                        self.image_preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )

                    self.image_preview.setPixmap(scaled_pixmap)
                    self.image_preview.setStyleSheet("border: 1px solid #555;")
                except Exception as e:
                    self.image_preview.setText(f"Hiba a kép betöltésekor: {e}")
                    self.image_preview.setStyleSheet("border: 1px dashed #777; background-color: #fcc;")

    def start_script(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())