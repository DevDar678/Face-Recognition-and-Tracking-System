import sys
from PyQt5.QtWidgets import QApplication, QTabWidget, QMainWindow, QVBoxLayout, QWidget
from face_register import FaceRegisterTab
from face_tracking_multiple import FaceTrackingTab
from step1_extract_faces import FaceExtractorApp
from performance import SystemMonitorGraph, PerformanceGraph
from PyQt5.QtCore import QTimer
import cv2
import time

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1200, 900)

        tabs = QTabWidget()
        
        # Create tabs
        tabs.addTab(FaceExtractorApp(), "Extract Faces")
        tabs.addTab(FaceRegisterTab(), "Register Faces")
        
        # Create face tracking tab
        self.face_tracking_tab = FaceTrackingTab()
        tabs.addTab(self.face_tracking_tab, "Track Faces")
        
        # Create performance tab
        performance_tab = QWidget()
        performance_layout = QVBoxLayout()
        
        # Add system monitor graph
        self.system_monitor = SystemMonitorGraph("System Resources")
        performance_layout.addWidget(self.system_monitor)
        
        # Add performance graph
        self.performance_graph = PerformanceGraph()
        performance_layout.addWidget(self.performance_graph)
        
        # Connect face tracking performance signals
        self.face_tracking_tab.performance_update.connect(self.update_performance_graphs)
        
        # Add update timer for system monitor
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.system_monitor.update_data)
        self.graph_timer.start(1000)  # Update every second
        
        performance_tab.setLayout(performance_layout)
        tabs.addTab(performance_tab, "Performance")
        
        self.setCentralWidget(tabs)

    def update_performance_graphs(self, total_faces, correct_matches):
        """Update performance graph with face recognition metrics"""
        self.performance_graph.update_accuracy(total_faces, correct_matches)
  


def get_main_app():
    return MainApp()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.setWindowTitle("Multi-Threaded Face Recognition")

    window.show()
    sys.exit(app.exec_())