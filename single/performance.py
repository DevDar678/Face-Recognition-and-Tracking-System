import os
import psutil
import time
from collections import deque
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt

class SystemMonitorGraph(QGraphicsView):
    def __init__(self, title="Face Recognition Resources", parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)  # Increased height to 300 pixels
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.title = title
        self.process = psutil.Process(os.getpid())
        
        self.cpu_data = deque(maxlen=100)    # Process CPU %
        self.memory_data = deque(maxlen=100) # Process memory in %
        self.max_value = 180  # Updated to 180% as the max value
        self.timestamps = deque(maxlen=100)
        
        # Get total system memory for percentage calculation
        self.total_system_memory = psutil.virtual_memory().total / (1024 ** 2)  # Convert to MB
        
        # Initialize CPU tracking variables
        self.last_cpu_times = self.process.cpu_times()
        self.last_update_time = time.time()
        
        self.draw_axes()
        
    def draw_axes(self):
        self.scene.clear()
        title = self.scene.addText(self.title)
        title.setDefaultTextColor(QColor(0, 0, 0))
        title.setPos(120, 5)
        
        black_pen = QPen(QColor(0, 0, 0))
        black_pen.setWidth(1)
        
        # Y-axis (0-180% for both CPU and Memory), extended to y=280
        self.scene.addLine(50, 40, 50, 280, black_pen)  # Extended to y=280
        self.scene.addLine(50, 280, 400, 280, black_pen)  # Updated bottom line
        
        # Y-axis markers for both CPU and Memory (now up to 180%)
        scale_factor = 240 / self.max_value  # 240 pixels / 180% = 1.333 pixels per %
        for i in [0, 30, 60, 90, 120, 150, 180]:
            y_pos = 280 - (i * scale_factor)  # Adjusted for new scale
            self.scene.addLine(45, y_pos, 50, y_pos, black_pen)
            self.scene.addText(f"{i}%").setPos(20, y_pos - 10)
        
    def update_data(self):
        try:
            # Get current CPU times and calculate usage
            current_cpu_times = self.process.cpu_times()
            now = time.time()
            time_elapsed = now - self.last_update_time
            
            # Calculate CPU percentage (total across all cores)
            cpu_time_elapsed = sum(current_cpu_times) - sum(self.last_cpu_times)
            cpu_percent = (cpu_time_elapsed / time_elapsed) * 100 if time_elapsed > 0 else 0
            
            # Update tracking variables
            self.last_cpu_times = current_cpu_times
            self.last_update_time = now
            
            # Get process memory (USS - Unique Set Size)
            mem = self.process.memory_full_info()
            mem_mb = mem.uss / (1024 ** 2)  # Convert to MB for calculation
            # Convert to percentage of total system memory
            mem_percent = (mem_mb / self.total_system_memory) * 100
            
            # Store data
            self.cpu_data.append(cpu_percent)
            self.memory_data.append(mem_percent)  # Storing percentage
            self.timestamps.append(now)
            
            self.update_graph()
        except Exception as e:
            print(f"Error in update_data: {str(e)}")
            # Fall back to simple CPU measurement
            self.cpu_data.append(self.process.cpu_percent(interval=0.1))
            mem = self.process.memory_full_info()
            mem_mb = mem.uss / (1024 ** 2)
            mem_percent = (mem_mb / self.total_system_memory) * 100
            self.memory_data.append(mem_percent)  # Store as percentage
            self.timestamps.append(time.time())
            self.update_graph()
        
    def update_graph(self):
        self.draw_axes()
        
        # Draw CPU graph
        cpu_pen = QPen(QColor(255, 0, 0), 2)  # Red
        for i in range(1, len(self.cpu_data)):
            x1 = 50 + (i - 1) * 3.5
            y1 = 280 - (self.cpu_data[i - 1] / self.max_value * 240)  # Adjusted for 0-180%
            x2 = 50 + i * 3.5
            y2 = 280 - (self.cpu_data[i] / self.max_value * 240)  # Adjusted for 0-180%
            # Clip y-values to ensure they stay within graph bounds
            y1 = max(40, min(280, y1))
            y2 = max(40, min(280, y2))
            self.scene.addLine(x1, y1, x2, y2, cpu_pen)
            
        # Draw Memory graph (now in %)
        mem_pen = QPen(QColor(0, 0, 255), 2)  # Blue
        for i in range(1, len(self.memory_data)):
            x1 = 50 + (i - 1) * 3.5
            y1 = 280 - (self.memory_data[i - 1] / self.max_value * 240)  # Adjusted for 0-180%
            x2 = 50 + i * 3.5
            y2 = 280 - (self.memory_data[i] / self.max_value * 240)  # Adjusted for 0-180%
            # Clip y-values to ensure they stay within graph bounds
            y1 = max(40, min(280, y1))
            y2 = max(40, min(280, y2))
            self.scene.addLine(x1, y1, x2, y2, mem_pen)
                
        # Add legends and current values
        self.scene.addText("CPU %").setPos(300, 30)
        self.scene.addRect(280, 30, 15, 15, QPen(Qt.black), QColor(255, 0, 0))
        self.scene.addText("Memory %").setPos(300, 50)  # Updated label
        self.scene.addRect(280, 50, 15, 15, QPen(Qt.black), QColor(0, 0, 255))
        
        if self.cpu_data:
            self.scene.addText(f"CPU: {self.cpu_data[-1]:.1f}%").setPos(250, 70)
        if self.memory_data:
            self.scene.addText(f"Mem: {self.memory_data[-1]:.2f}%").setPos(250, 90)  # Updated to show %




class PerformanceGraph(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 200)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.accuracy_data = deque(maxlen=100)
        self.fps_data = deque(maxlen=100)
        self.frame_times = deque(maxlen=10)  # Store last 10 frame times for smoothing
        self.last_update_time = time.time()
        
        self.draw_axes()
        
    def draw_axes(self):
        self.scene.clear()
        title = self.scene.addText("Face Recognition Performance")
        title.setDefaultTextColor(QColor(0, 0, 0))
        title.setPos(120, 5)
        
        black_pen = QPen(QColor(0, 0, 0))
        black_pen.setWidth(1)
        
        self.scene.addLine(50, 40, 50, 180, black_pen)
        self.scene.addLine(50, 180, 400, 180, black_pen)
        
        self.scene.addText("Accuracy %").setPos(5, 20)
        self.scene.addText("Time").setPos(350, 190)
        
        for i in range(0, 101, 20):
            y_pos = 180 - (i * 1.4)
            self.scene.addLine(45, y_pos, 50, y_pos, black_pen)
            self.scene.addText(f"{i}%").setPos(20, y_pos - 10)
        
    def update_accuracy(self, total_faces, correct_matches):
        if total_faces > 0:
            accuracy = (correct_matches / total_faces) * 100
        else:
            accuracy = 0
            
        self.accuracy_data.append(accuracy)
        
        # Calculate FPS using smoothed average of frame times
        current_time = time.time()
        frame_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.frame_times.append(frame_time)
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
        
        self.fps_data.append(fps)
        self.update_graph()
        
    def update_graph(self):
        self.draw_axes()
        
        acc_pen = QPen(QColor(0, 255, 0))  # Green for accuracy
        acc_pen.setWidth(2)
        fps_pen = QPen(QColor(255, 165, 0))  # Orange for FPS
        fps_pen.setWidth(2)
        
        for i in range(1, len(self.accuracy_data)):
            x1 = 50 + (i-1) * 3.5
            y1 = 180 - (self.accuracy_data[i-1] * 1.4)
            x2 = 50 + i * 3.5
            y2 = 180 - (self.accuracy_data[i] * 1.4)
            self.scene.addLine(x1, y1, x2, y2, acc_pen)
            
        if self.fps_data:
            # Scale FPS to fit the 0-100% range
            max_fps = max(max(self.fps_data), 30)  # Use either max observed or 30 as baseline
            for i in range(1, len(self.fps_data)):
                x1 = 50 + (i-1) * 3.5
                y1 = 180 - ((self.fps_data[i-1]/max_fps * 100) * 1.4)
                x2 = 50 + i * 3.5
                y2 = 180 - ((self.fps_data[i]/max_fps * 100) * 1.4)
                self.scene.addLine(x1, y1, x2, y2, fps_pen)
                
        self.scene.addText("Accuracy").setPos(300, 30)
        self.scene.addRect(280, 30, 15, 15, QPen(Qt.black), QColor(0, 255, 0))
        self.scene.addText("FPS").setPos(300, 50)
        self.scene.addRect(280, 50, 15, 15, QPen(Qt.black), QColor(255, 165, 0))
        
        if self.accuracy_data:
            self.scene.addText(f"Accuracy: {self.accuracy_data[-1]:.1f}%").setPos(250, 70)
        if self.fps_data:
            # Display the actual FPS value (not scaled)
            self.scene.addText(f"FPS: {self.fps_data[-1]:.1f}").setPos(250, 90)