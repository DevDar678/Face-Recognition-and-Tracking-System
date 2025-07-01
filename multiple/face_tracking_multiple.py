import cv2
import sqlite3
import numpy as np
import face_recognition
import time
from PyQt5.QtWidgets import (QWidget, QGridLayout, QPushButton, QFileDialog, 
                             QLabel, QVBoxLayout)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class FaceLoaderThread(QThread):
    finished = pyqtSignal(list, list)
    
    def run(self):
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
        
        self.finished.emit(encodings, names)


class VideoProcessorThread(QThread):
    frame_processed = pyqtSignal(int, np.ndarray)
    performance_data = pyqtSignal(int, int, float)  # total_faces, correct_matches, processing_time

    def __init__(self, index, frame, known_encodings, known_names):
        super().__init__()
        self.index = index
        self.frame = frame
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.start_time = time.time()
    
    def run(self):
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, boxes)

        correct_matches = 0
        total_faces = len(boxes)

        for (box, enc) in zip(boxes, encs):
            matches = face_recognition.compare_faces(self.known_encodings, enc, tolerance=0.5)
            name = "Unknown"
            box_color = (255, 0,0) 

            if True in matches:
                correct_matches += 1
                counts = {}
                for i, matched in enumerate(matches):
                    if matched:
                        counts[self.known_names[i]] = counts.get(self.known_names[i], 0) + 1
                name = max(counts, key=counts.get)
                box_color = (0, 255,0) 

            top, right, bottom, left = box
            cv2.rectangle(rgb, (left, top), (right, bottom), box_color, 2)
            cv2.putText(rgb, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        processing_time = time.time() - self.start_time
        self.frame_processed.emit(self.index, rgb)
        self.performance_data.emit(total_faces, correct_matches, processing_time)


class FaceTrackingTab(QWidget):
    performance_update = pyqtSignal(int, int, float)  # total_faces, correct_matches, processing_time

    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.video_widgets = []
        self.videos = []
        self.timers = []
        self.processing_threads = []
        self.video_start_times = [None] * 4  # Track start times for each video
        self.video_durations = [0] * 4  # Track durations for each video

        self.layout.setSpacing(15)
        self.layout.setContentsMargins(15, 15, 15, 15)

        for i in range(4):
            video_container = QVBoxLayout()
            label = QLabel(f"Video Feed {i+1}")
            label.setFixedSize(400, 300)
            label.setStyleSheet("""
                border: 2px solid #000328;
                border-radius: 5px;
                background-color: #f0f0f0;
                qproperty-alignment: AlignCenter;
            """)
            video_container.addWidget(label)

            btn = QPushButton(f"Load Video {i+1}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #000328;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #000112;
                }
            """)
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda _, idx=i: self.load_video(idx))
            video_container.addWidget(btn)

            status_label = QLabel("No video loaded")
            status_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
            video_container.addWidget(status_label)

            self.layout.addLayout(video_container, i // 2, i % 2)

            self.video_widgets.append({
                'display': label,
                'status': status_label
            })
            self.videos.append(None)
            self.timers.append(QTimer())
            self.timers[i].timeout.connect(lambda idx=i: self.update_frame(idx))
            self.processing_threads.append(None)

        self.known_encodings = []
        self.known_names = []
        self.face_loader_thread = FaceLoaderThread()
        self.face_loader_thread.finished.connect(self.on_faces_loaded)
        self.face_loader_thread.start()
        self.video_widgets[0]['status'].setText("Loading known faces...")

    def on_faces_loaded(self, encodings, names):
        self.known_encodings = encodings
        self.known_names = names
        self.video_widgets[0]['status'].setText(f"Loaded {len(names)} known faces")

    def load_video(self, index):
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
            self.video_start_times[index] = time.time()  # Record start time when video loads
            self.timers[index].start(30)

    def update_frame(self, index):
        cap = self.videos[index]
        if cap is None:
            return

        ret, frame = cap.read()
        if not ret:
            self.timers[index].stop()
            duration = time.time() - self.video_start_times[index]  # Calculate duration
            self.video_durations[index] = duration
            self.video_widgets[index]['status'].setText(
                f"Video ended in {duration:.2f} seconds"
            )
            self.video_widgets[index]['status'].setStyleSheet("color: #e74c3c;")
            return

        if self.processing_threads[index] and self.processing_threads[index].isRunning():
            return

        self.processing_threads[index] = VideoProcessorThread(
            index, frame, self.known_encodings, self.known_names
        )
        self.processing_threads[index].frame_processed.connect(self.display_processed_frame)
        self.processing_threads[index].performance_data.connect(self.handle_performance_data)
        self.processing_threads[index].start()

    def display_processed_frame(self, index, rgb_frame):
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            400, 300,
            aspectRatioMode=1,
            transformMode=1
        )
        self.video_widgets[index]['display'].setPixmap(pixmap)

    def handle_performance_data(self, total_faces, correct_matches, processing_time):
        if total_faces > 0:
            self.performance_update.emit(total_faces, correct_matches, processing_time)

    def closeEvent(self, event):
        for thread in self.processing_threads:
            if thread and thread.isRunning():
                thread.terminate()
        if self.face_loader_thread.isRunning():
            self.face_loader_thread.terminate()
        event.accept()