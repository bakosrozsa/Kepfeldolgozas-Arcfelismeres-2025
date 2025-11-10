import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QTextEdit, QFileDialog, QHBoxLayout, QLineEdit
)
from PyQt6.QtCore import QProcess, Qt


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arcfelismerés")
        self.setMinimumSize(600, 400)

        # --- Layout ---
        main_layout = QVBoxLayout()

        # Fájl kiválasztás
        file_layout = QHBoxLayout()
        self.file_path_display = QLineEdit()
        self.file_path_display.setPlaceholderText("Válassz egy képet...")
        browse_btn = QPushButton("Tallózás")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path_display)
        file_layout.addWidget(browse_btn)
        main_layout.addLayout(file_layout)

        # Futtatás gomb
        self.run_btn = QPushButton("Script Indítása")
        self.run_btn.setFixedHeight(40)
        self.run_btn.clicked.connect(self.start_script)
        self.run_btn.setEnabled(False)
        main_layout.addWidget(self.run_btn)

        # Log ablak
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Itt jelenik meg a script kimenete")
        main_layout.addWidget(QLabel("Folyamat Log:"))
        main_layout.addWidget(self.log_output)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.process = QProcess(self)

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Kép megnyitása", "", "Image Files (*.png *.jpg *.bmp)")
        if fname:
            self.file_path_display.setText(fname)
            self.run_btn.setEnabled(True)

    def start_script(self):
        # todo: script hívása
        pass;


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())