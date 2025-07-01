import cv2
import sqlite3
import numpy as np
import time
import face_recognition
from PyQt5.QtWidgets import (QWidget, QGridLayout, QPushButton, QFileDialog, 
                            QLabel, QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import time

class FaceTrackingTab(QWidget):
    def __init__(self, performance_graph=None, system_monitor=None):
        super().__init__()
        self.performance_graph = performance_graph
        self.system_monitor = system_monitor
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.video_widgets = []
        self.videos = []
        self.timers = []
        self.frame_count = 0
        self.last_update_time = 0

        self.video_start_times = [None] * 4
        self.video_durations = [0] * 4

        # Set up UI
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(15, 15, 15, 15)

        # Create video feeds
        for i in range(4):
            video_container = QVBoxLayout()
            video_container.setSpacing(10)

            # Video display label
            label = QLabel(f"Video Feed {i+1}")
            label.setFixedSize(400, 300)
            label.setStyleSheet("""
                border: 2px solid #000328;
                border-radius: 5px;
                background-color: #f0f0f0;
                qproperty-alignment: AlignCenter;
            """)
            video_container.addWidget(label)

            # Load video button
            btn = QPushButton(f"Load Video {i+1}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #000328;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 16px
                }
                QPushButton:hover {
                    background-color: #000112;
                }
            """)
            btn.setFixedHeight(40)
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _, idx=i: self.load_video(idx))
            video_container.addWidget(btn)

            # Status label
            status_label = QLabel("No video loaded")
            status_label.setStyleSheet("color: #7f8c8d; font-style: italic; font-weight: bold; font-size: 14px")
            video_container.addWidget(status_label)

            self.layout.addLayout(video_container, i // 2, i % 2)

            self.video_widgets.append({
                'display': label,
                'status': status_label
            })
            self.videos.append(None)
            self.timers.append(QTimer())
            self.timers[i].timeout.connect(lambda idx=i: self.update_frame(idx))

        # Load known faces from database
        self.known_encodings, self.known_names = self.load_known_faces()

        # Initialize frame timing for FPS calculation
        self.frame_times = []
        self.last_frame_time = 0

    def load_known_faces(self):
        """Load known face encodings from database"""
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name, encoding FROM faces")
        results = cursor.fetchall()
        conn.close()

        names = []
        encodings = []
        for name, blob in results:
            names.append(name)
            encodings.append(np.frombuffer(blob, dtype=np.float64))

        return encodings, names

    def load_video(self, index):
        """Load a video file into the specified video slot"""
        file, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select Video {index+1}", 
            "", 
            "Videos (*.mp4 *.avi *.mov)"
        )
      
        if file:
            self.videos[index] = cv2.VideoCapture(file)
            self.video_widgets[index]['status'].setText(f"Loaded: {file.split('/')[-1]}")
            self.video_widgets[index]['status'].setStyleSheet("color: #27ae60;")
            self.video_start_times[index] = time.time()
            self.timers[index].start(30)
            
            # Reset performance counters
            if self.performance_graph:
                self.performance_graph.accuracy_data.clear()
                self.performance_graph.fps_data.clear()
            if self.system_monitor:
                self.system_monitor.cpu_data.clear()
                self.system_monitor.memory_data.clear()

    def update_frame(self, index):
        """Process and display the next video frame"""
        cap = self.videos[index]
        if cap is None:
            return

        ret, frame = cap.read()
        if not ret:
            self.timers[index].stop()
            duration = time.time() - self.video_start_times[index]
            self.video_durations[index] = duration
            self.video_widgets[index]['status'].setText(
                f"Video ended in {duration:.2f} seconds"
            )
            self.video_widgets[index]['status'].setStyleSheet("color: #e74c3c;")
            return

        # Update system monitor
        if self.system_monitor:
            self.system_monitor.update_data()

        # Calculate FPS
        current_time = time.time()
        if self.last_frame_time > 0:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 10:
                self.frame_times.pop(0)
        self.last_frame_time = current_time

        # Process frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, boxes)

        # Face recognition
        correct_matches = 0
        total_faces = len(encs)
        for (box, enc) in zip(boxes, encs):
            matches = face_recognition.compare_faces(self.known_encodings, enc, tolerance=0.5)
            name = "Unknown"
            box_color = (255, 0,0)  

            if True in matches:
                counts = {}
                for i, matched in enumerate(matches):
                    if matched:
                        counts[self.known_names[i]] = counts.get(self.known_names[i], 0) + 1
                name = max(counts, key=counts.get)
                correct_matches += 1
                box_color = (0, 255, 0)  

            # Draw bounding box and name
            top, right, bottom, left = box
            cv2.rectangle(rgb, (left, top), (right, bottom), box_color, 2)
            cv2.putText(rgb, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Update performance metrics
        if self.performance_graph:
            # Calculate accuracy
            accuracy = (correct_matches / total_faces) * 100 if total_faces > 0 else 0
            self.performance_graph.accuracy_data.append(accuracy)
            
            # Calculate FPS
            if len(self.frame_times) > 0:
                fps = 1 / (sum(self.frame_times) / len(self.frame_times))
                self.performance_graph.fps_data.append(fps)
            
            self.performance_graph.update_graph()

        # Display frame
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(400, 300, aspectRatioMode=1, transformMode=1)
        self.video_widgets[index]['display'].setPixmap(pixmap)

        # Increment frame counter
        self.frame_count += 1