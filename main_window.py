import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QTextEdit, QFileDialog, QHBoxLayout, QLineEdit, QFormLayout
)
from PyQt6.QtCore import QProcess, Qt


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arcfelismerés")
        self.setMinimumSize(600, 500)

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

        # -- script indítása button --
        self.run_btn = QPushButton("Script Indítása")
        self.run_btn.setFixedHeight(50)
        self.run_btn.setStyleSheet(
            "font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.run_btn.clicked.connect(self.start_script)
        self.run_btn.setEnabled(False)
        main_layout.addWidget(self.run_btn)

        # -- Log --
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_output)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # -- Tallózó függvény --
    def browse_file(self, target_input, window_title, file_filter):
        fname, _ = QFileDialog.getOpenFileName(self, window_title, "", file_filter)
        if fname:
            target_input.setText(fname)

    def start_script(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())