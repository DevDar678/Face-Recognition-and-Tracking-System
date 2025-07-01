import sys
import os
import cv2
import face_recognition
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QProgressBar, QSpinBox,
                            QMessageBox, QGroupBox, QLineEdit, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QStyleFactory

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
                background-color: #000112;
                color: #666666;
            }
            QPushButton#cancelButton {
                background-color: red;
                color: white;
                font-weight: bold;
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
                font-size: 16px;
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

        # Variables for face extraction
        self.video_capture = None
        self.total_frames = 0
        self.current_frame = 0
        self.face_count = 0
        self.output_dir = ""
        self.interval = 10
        self.is_extracting = False
        self.extraction_timer = QTimer()
        self.extraction_timer.timeout.connect(self.process_next_frame)

        self.create_ui()

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
        self.output_dir = self.output_dir_edit.text()
        self.interval = self.interval_spin.value()

        if not video_path:
            QMessageBox.warning(self, "Error", "Please select a video file")
            return
            
        if not self.output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory")
            return

        # Initialize video capture
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            QMessageBox.warning(self, "Error", "Could not open video file")
            return

        # Prepare for extraction
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.face_count = 0
        self.is_extracting = True

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Update UI
        self.extract_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Start processing frames
        self.extraction_timer.start(0)  # Process as fast as possible

    def process_next_frame(self):
        if not self.is_extracting or self.current_frame >= self.total_frames:
            self.extraction_finished()
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.extraction_finished()
            return

        # Process frame if it's an interval frame
        if self.current_frame % self.interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            
            for (top, right, bottom, left) in boxes:
                if not self.is_extracting:
                    break
                face_img = frame[top:bottom, left:right]
                face_path = os.path.join(self.output_dir, f"face_{self.face_count}.jpg")
                cv2.imwrite(face_path, face_img)
                self.face_count += 1

        self.current_frame += 1
        progress = int((self.current_frame / self.total_frames) * 100)
        self.progress_bar.setValue(progress)

        # Allow the GUI to process events between frames
        QApplication.processEvents()

    def cancel_extraction(self):
        self.is_extracting = False
        self.extraction_timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.reset_ui()
        QMessageBox.information(self, "Cancelled", "Face extraction was cancelled.")

    def extraction_finished(self):
        self.extraction_timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        if self.is_extracting:  # Only show success if not cancelled
            self.reset_ui()
            QMessageBox.information(
                self, "Success", 
                f"Face extraction completed!\n{self.face_count} faces were extracted."
            )
        self.is_extracting = False

    def reset_ui(self):
        self.extract_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def closeEvent(self, event):
        if self.is_extracting:
            self.cancel_extraction()
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