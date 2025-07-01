# performance_tab.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from performance import SystemMonitorGraph, PerformanceGraph  # Import both graph classes
from PyQt5.QtCore import QTimer

class PerformanceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Create system monitor graph
        self.system_monitor = SystemMonitorGraph()
        self.layout.addWidget(self.system_monitor)
        
        # Create performance graph
        self.performance_graph = PerformanceGraph()
        self.layout.addWidget(self.performance_graph)
        
        # Timer to update system monitor
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.system_monitor.update_data)
        self.monitor_timer.start(1000)  # Update every second
        
    def get_performance_graph(self):
        return self.performance_graph