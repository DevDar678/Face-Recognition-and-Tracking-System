# main.py
import sys
from PyQt5.QtWidgets import QApplication, QTabWidget, QMainWindow
from face_register import FaceRegisterTab
from face_tracking import FaceTrackingTab
from step1_extract_faces import FaceExtractorApp
from performance_tab import PerformanceTab
import cv2
import time

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single-Threaded Face Recognition")
        self.setGeometry(100, 100, 1200, 900)  # Increased size for more graphs

        tabs = QTabWidget()
        
        # Create performance tab with both graphs
        self.performance_tab = PerformanceTab()
        
        # Create face tracking tab with references to both graphs
        tabs.addTab(FaceExtractorApp(), "Extract Faces")
        tabs.addTab(FaceRegisterTab(), "Register Faces")
        tabs.addTab(FaceTrackingTab(
            performance_graph=self.performance_tab.get_performance_graph(),
            system_monitor=self.performance_tab.system_monitor
        ), "Track Faces")
        tabs.addTab(self.performance_tab, "Performance Metrics")

        self.setCentralWidget(tabs)
        print("[single/main.py] Starting Single-Threaded App")

   

        
def get_main_app():
    return MainApp()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = get_main_app()
    window.setWindowTitle("Single-Threaded Face Recognition")

    window.show()
    sys.exit(app.exec_())