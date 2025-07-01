import os
import shutil
import sqlite3
import numpy as np
import face_recognition
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFileDialog, 
                            QLabel, QLineEdit, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

DB_PATH = "faces.db"
REFERENCE_IMAGE_DIR = "reference_images"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

class AutoLabelWorker(QThread):
    update_status = pyqtSignal(str, str)  # message, color
    finished = pyqtSignal(int, str)  # matches, name
    error_occurred = pyqtSignal(str)

    def __init__(self, name, ref_img_paths, faces_folder):
        super().__init__()
        self.name = name
        self.ref_img_paths = ref_img_paths
        self.faces_folder = faces_folder
        self._is_running = True

    def run(self):
        try:
            ref_encs = []
            for path in self.ref_img_paths:
                if not self._is_running:
                    return
                
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    ref_encs.append(encs[0])
                self.update_status.emit(f"Processing reference image: {os.path.basename(path)}", "blue")

            if not ref_encs:
                self.update_status.emit("No clear face found in reference images. Using fallback matching.", "orange")

            matches = 0
            kept_reference = False
            face_files = os.listdir(self.faces_folder)
            total_files = len(face_files)

            for i, fname in enumerate(face_files):
                if not self._is_running:
                    return
                    
                fpath = os.path.join(self.faces_folder, fname)
                try:
                    img = face_recognition.load_image_file(fpath)
                    encs = face_recognition.face_encodings(img)
                    if not encs:
                        continue
                    
                    candidate_enc = encs[0]

                    match = False
                    if ref_encs:
                        for ref_enc in ref_encs:
                            dist = np.linalg.norm(ref_enc - candidate_enc)
                            if dist < 0.5:
                                match = True
                                break
                    else:
                        match = True if matches < 20 else False

                    if match:
                        self.save_encoding_to_db(self.name, candidate_enc)
                        matches += 1

                        if not kept_reference:
                            dest_path = os.path.join(REFERENCE_IMAGE_DIR, f"{self.name}.jpg")
                            shutil.copy(fpath, dest_path)
                            kept_reference = True

                        os.remove(fpath)

                    # Update progress every 10 files or for the last file
                    if i % 10 == 0 or i == total_files - 1:
                        self.update_status.emit(f"Processed {i+1}/{total_files} files, found {matches} matches", "blue")

                except Exception as e:
                    self.update_status.emit(f"Error processing {fname}: {str(e)}", "red")
                    continue
                
            if matches > 0:
                self.finished.emit(matches, self.name)
            else:
                self.update_status.emit("No matching faces found", "red")

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self._is_running = False
        self.wait()

    def save_encoding_to_db(self, name, encoding):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding.tobytes()))
        conn.commit()
        conn.close()

class FaceRegisterTab(QWidget):
    def __init__(self):
        super().__init__()
        init_db()

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Name input section
        name_layout = QVBoxLayout()
        name_label = QLabel("Person's Name:")
        name_label.setStyleSheet("font-weight: bold; font-size: 20px;")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter person's name (e.g. ali)")
        self.name_input.setMinimumHeight(40)
        self.name_input.setStyleSheet("font-size: 18px;")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # Reference images section
        ref_layout = QVBoxLayout()
        ref_label = QLabel("Reference Images:")
        ref_label.setStyleSheet("font-weight: bold; font-size: 20px;")
        self.ref_status = QLabel("No reference images selected")
        self.ref_status.setStyleSheet("color: #666; font-style: italic; font-size: 16px;")
        
        self.select_ref_btn = QPushButton("Select 2-3 Reference Images")
        self.select_ref_btn.setMinimumHeight(40)
        self.select_ref_btn.setStyleSheet("font-size: 18px;")
        self.select_ref_btn.clicked.connect(self.select_reference_images)
        
        ref_layout.addWidget(ref_label)
        ref_layout.addWidget(self.select_ref_btn)
        ref_layout.addWidget(self.ref_status)
        layout.addLayout(ref_layout)

        # Extracted faces folder section
        faces_layout = QVBoxLayout()
        faces_label = QLabel("Extracted Faces Folder:")
        faces_label.setStyleSheet("font-weight: bold; font-size: 20px;")
        self.faces_status = QLabel("No folder selected")
        self.faces_status.setStyleSheet("color: #666; font-style: italic; font-size: 16px;")
        
        self.select_faces_btn = QPushButton("Select Extracted Faces Folder")
        self.select_faces_btn.setMinimumHeight(40)
        self.select_faces_btn.setStyleSheet("font-size: 18px")
        self.select_faces_btn.clicked.connect(self.select_faces_folder)
        
        faces_layout.addWidget(faces_label)
        faces_layout.addWidget(self.select_faces_btn)
        faces_layout.addWidget(self.faces_status)
        layout.addLayout(faces_layout)

        # Start button section
        self.start_btn = QPushButton("START PROCESS")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("background-color: #000328; color: white; font-weight: bold; font-size: 20px;")
        self.start_btn.clicked.connect(self.start_auto_label)
        layout.addWidget(self.start_btn)

        # Cancel button (hidden by default)
        self.cancel_btn = QPushButton("CANCEL PROCESS")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.cancel_btn.clicked.connect(self.cancel_auto_label)
        self.cancel_btn.hide()
        layout.addWidget(self.cancel_btn)

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; margin-top: 20px; font-size: 22px;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self.ref_img_paths = []
        self.faces_folder = None
        self.worker_thread = None

        if not os.path.exists(REFERENCE_IMAGE_DIR):
            os.makedirs(REFERENCE_IMAGE_DIR)

    def select_reference_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select 2-3 Reference Images", "", "Images (*.jpg *.jpeg *.png)")
        if files:
            if len(files) > 3:
                QMessageBox.warning(self, "Too Many Images", "Please select only up to 3 reference images.")
                return
            
            self.ref_img_paths = files
            self.ref_status.setText(f"{len(files)} image(s) selected")
            self.ref_status.setStyleSheet("color: green;")
            QMessageBox.information(self, "Success", "Reference images uploaded successfully!")

    def select_faces_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Faces Folder")
        if folder:
            self.faces_folder = folder
            self.faces_status.setText(f"Folder selected: {os.path.basename(folder)}")
            self.faces_status.setStyleSheet("color: green;")
            QMessageBox.information(self, "Success", "Faces folder uploaded successfully!")

    def validate_inputs(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Missing Information", "Please enter the person's name.")
            return False
            
        if not self.ref_img_paths:
            QMessageBox.warning(self, "Missing Information", "Please select reference images.")
            return False
            
        if not self.faces_folder:
            QMessageBox.warning(self, "Missing Information", "Please select the extracted faces folder.")
            return False
            
        return True

    def start_auto_label(self):
        if not self.validate_inputs():
            return
            
        person_name = self.name_input.text().strip()
        
        # Disable UI elements during processing
        self.name_input.setEnabled(False)
        self.select_ref_btn.setEnabled(False)
        self.select_faces_btn.setEnabled(False)
        self.start_btn.hide()
        self.cancel_btn.show()
        
        # Create and start worker thread
        self.worker_thread = AutoLabelWorker(
            person_name, 
            self.ref_img_paths, 
            self.faces_folder
        )
        self.worker_thread.update_status.connect(self.update_status)
        self.worker_thread.finished.connect(self.on_auto_label_finished)
        self.worker_thread.error_occurred.connect(self.on_auto_label_error)
        self.worker_thread.start()

    def cancel_auto_label(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
        self.reset_ui()
        self.status_label.setText("Status: Process canceled")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

    def update_status(self, message, color):
        self.status_label.setText(f"Status: {message}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def on_auto_label_finished(self, matches, name):
        self.reset_ui()
        self.status_label.setText(f"Status: Saved {matches} encodings for {name}")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        QMessageBox.information(self, "Success", f"Process completed successfully! Saved {matches} encodings.")

    def on_auto_label_error(self, error_msg):
        self.reset_ui()
        self.status_label.setText("Status: Error occurred")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")

    def reset_ui(self):
        self.name_input.setEnabled(True)
        self.select_ref_btn.setEnabled(True)
        self.select_faces_btn.setEnabled(True)
        self.start_btn.show()
        self.cancel_btn.hide()
        if self.worker_thread:
            self.worker_thread = None

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Process Running',
                "Auto-labeling is still running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def delete_user_completely(name):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete all entries for this name
        cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
        
        # Reset autoincrement counter if you want to reuse IDs
        # cursor.execute("DELETE FROM sqlite_sequence WHERE name='faces'")
        
        conn.commit()
        conn.close()