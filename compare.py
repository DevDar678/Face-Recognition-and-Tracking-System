# compare.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout
from PyQt5.QtWidgets import QMainWindow

# Add both directories to sys.path
sys.path.append(os.path.abspath("single"))
sys.path.append(os.path.abspath("multiple"))

# Import the MainApp factories
from single.main import get_main_app as get_single_app
from multiple.main import get_main_app as get_multi_app

class ComparisonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single vs Multithreaded Face Recognition Comparison")
        self.setGeometry(100, 100, 2500, 1000)  # Wide layout

        layout = QHBoxLayout()

        # Get both apps
        self.single_app_widget = get_single_app()
        self.multi_app_widget = get_multi_app()

        # Add their central widgets to a layout
        layout.addWidget(self.single_app_widget.centralWidget())
        layout.addWidget(self.multi_app_widget.centralWidget())

        # Wrap layout in a QWidget and set as central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ComparisonWindow()
    window.show()
    sys.exit(app.exec_())
