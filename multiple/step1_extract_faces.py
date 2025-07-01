import sys
import os
import cv2
import face_recognition
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QProgressBar, QSpinBox,
                            QMessageBox, QGroupBox, QLineEdit, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QStyleFactory

class ExtractionWorker(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, output_dir, interval):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval = interval
        self._is_running = True

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_id = 0
            face_id = 0

            while self._is_running and frame_id < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % self.interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes = face_recognition.face_locations(rgb)
                    
                    for (top, right, bottom, left) in boxes:
                        if not self._is_running:
                            break
                        face_img = frame[top:bottom, left:right]
                        face_path = os.path.join(self.output_dir, f"face_{face_id}.jpg")
                        cv2.imwrite(face_path, face_img)
                        face_id += 1

                frame_id += 1
                progress = int((frame_id / total_frames) * 100)
                self.update_progress.emit(progress)

            cap.release()
            if self._is_running:
                self.finished.emit(face_id)
            else:
                self.finished.emit(0)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self._is_running = False
        self.wait()

class FaceExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Extractor")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                font-size: 20px;
                
            }
            QPushButton {
                background-color: #000328;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                min-width: 120px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #000112;
            }
            QPushButton:disabled {
                background-color: red;
                color: #666666;
            }
            QPushButton#cancelButton {
                background-color: #red;
            }
            QPushButton#cancelButton:hover {
                background-color: #662211;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                font-size: 14px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
            QLabel {
                font-size: 16px;
            }
            QSpinBox {
                padding: 5px;
                font-size: 16px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(20, 20, 20, 20)

        self.create_ui()
        self.extraction_worker = None

    def create_ui(self):
        # Video Selection Group
        video_group = QGroupBox("Video Selection")
        video_layout = QVBoxLayout()
        
        # Video path input
        path_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Select video file...")
        path_layout.addWidget(self.video_path_edit)
        
        browse_video_btn = QPushButton("Browse Video")
        browse_video_btn.clicked.connect(self.browse_video)
        path_layout.addWidget(browse_video_btn)
        
        video_layout.addLayout(path_layout)
        
        # Preview section
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #000;
                border: 1px solid #ddd;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        
        preview_frame.setLayout(preview_layout)
        video_layout.addWidget(preview_frame)
        
        video_group.setLayout(video_layout)
        self.layout.addWidget(video_group)

        # Output Group
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        # Output Directory
        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        dir_layout.addWidget(self.output_dir_edit)
        
        browse_dir_btn = QPushButton("Browse Output")
        browse_dir_btn.clicked.connect(self.browse_output)
        dir_layout.addWidget(browse_dir_btn)
        output_layout.addLayout(dir_layout)
        
        # Frame Interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Extract every:"))
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(10)
        interval_layout.addWidget(self.interval_spin)
        interval_layout.addWidget(QLabel("frames"))
        interval_layout.addStretch()
        output_layout.addLayout(interval_layout)
        
        output_group.setLayout(output_layout)
        self.layout.addWidget(output_group)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Progress: %p%")
        self.layout.addWidget(self.progress_bar)

        # Action Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.extract_btn = QPushButton("Start Extraction")
        self.extract_btn.clicked.connect(self.start_extraction)
        button_layout.addWidget(self.extract_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelButton")
        self.cancel_btn.clicked.connect(self.cancel_extraction)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        self.layout.addLayout(button_layout)

        # Add some spacing at the bottom
        self.layout.addStretch()

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.video_path_edit.setText(file_path)
            self.show_video_preview(file_path)

    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def show_video_preview(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.preview_label.setPixmap(pixmap.scaled(
                    self.preview_label.width(), 
                    self.preview_label.height(), 
                    Qt.KeepAspectRatio
                ))
            cap.release()

    def start_extraction(self):
        video_path = self.video_path_edit.text()
        output_dir = self.output_dir_edit.text()
        interval = self.interval_spin.value()

        if not video_path:
            QMessageBox.warning(self, "Error", "Please select a video file")
            return
            
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory")
            return

        # Disable UI elements during extraction
        self.extract_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.video_path_edit.setEnabled(False)
        self.output_dir_edit.setEnabled(False)
        self.interval_spin.setEnabled(False)
        self.progress_bar.setValue(0)

        # Create and start the worker thread
        self.extraction_worker = ExtractionWorker(video_path, output_dir, interval)
        self.extraction_worker.update_progress.connect(self.update_progress)
        self.extraction_worker.finished.connect(self.extraction_finished)
        self.extraction_worker.error_occurred.connect(self.show_error)
        self.extraction_worker.start()

    def cancel_extraction(self):
        if self.extraction_worker and self.extraction_worker.isRunning():
            self.extraction_worker.stop()
        self.reset_ui()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def extraction_finished(self, face_count):
        self.reset_ui()
        QMessageBox.information(
            self, "Success", 
            f"Face extraction completed!\n{face_count} faces were extracted."
        )

    def show_error(self, error_msg):
        self.reset_ui()
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")

    def reset_ui(self):
        self.extract_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.video_path_edit.setEnabled(True)
        self.output_dir_edit.setEnabled(True)
        self.interval_spin.setEnabled(True)
        if hasattr(self, 'extraction_worker'):
            self.extraction_worker = None

    def closeEvent(self, event):
        if hasattr(self, 'extraction_worker') and self.extraction_worker and self.extraction_worker.isRunning():
            reply = QMessageBox.question(
                self, 'Extraction in Progress',
                "Extraction is still running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_extraction()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # Set a modern font
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = FaceExtractorApp()
    window.show()
    sys.exit(app.exec_())