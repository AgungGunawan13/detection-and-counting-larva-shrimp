import sys
from ultralytics import YOLO
import cv2
import psutil
import GPUtil
import time
from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_Dialog


# =========================================
# THREAD KAMERA (BACKGROUND)
# =========================================
class CameraThread(QtCore.QThread):
    # Emit frame, inference time string, and updated count
    frame_ready = QtCore.pyqtSignal(object, str, int)

    def __init__(self, camera_index=1, raw_line_y=300):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        
        # Load local YOLO model
        self.model = YOLO("best.pt")
        
        self.line_y = raw_line_y 
        self.count = 0
        
        # Dictionary to store tracking IDs to avoid double counting
        self.tracked_objects = {}

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("Kamera gagal dibuka")
            return

        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Start timing inference
            start_time = time.time()
            
            # (Fix 2) Predict with YOLO GPU & Tracker Config parameter anti bayangan semu
            results = self.model.track(frame, conf=0.55, device=0, tracker="botsort.yaml", persist=True, verbose=False)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000 # in ms
            inf_time_str = f"{inference_time:.2f} ms"

            # Process tracked objects
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Draw bounding box and center point
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    
                    # Check for line crossing (Simple Y-crossing logic)
                    if track_id not in self.tracked_objects:
                        self.tracked_objects[track_id] = cy
                    else:
                        prev_cy = self.tracked_objects[track_id]
                        
                        # Only count top to bottom, or bottom to top crossing once
                        if (prev_cy < self.line_y and cy >= self.line_y) or (prev_cy > self.line_y and cy <= self.line_y):
                            self.count += 1
                            # Set it to the exact line to avoid multiple counts on same frame
                            self.tracked_objects[track_id] = cy 
                            
                        # Update position
                        self.tracked_objects[track_id] = cy

            self.frame_ready.emit(frame, inf_time_str, self.count)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# =========================================
# MAIN UI
# =========================================
class CameraApp(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.label_TENTANG.installEventFilter(self)
        self.setFixedSize(789, 716) # Mengunci UI agar ukurannya tidak bisa ditarik/dirusak kordinatnya




        # =====================================
        # VIEW KAMERA
        # =====================================
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        layout = QtWidgets.QVBoxLayout(self.ui.view_kamera)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        self.camera_thread = None

        # =====================================
        # LABEL PERSENTASE (AWAL)
        # =====================================
        self.ui.PERSENTASE_CPU.setText("0%")
        self.ui.PERSENTASE_GPU.setText("0%")

        # =====================================
        # TIMER CPU & GPU (BELUM DIMULAI)
        # =====================================
        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.timeout.connect(self.update_usage)

        # Button
        self.ui.MULAI.clicked.connect(self.start_camera)
        self.ui.SELESAI.clicked.connect(self.stop_camera)
        self.ui.CLOSE.clicked.connect(self.close)

    # =====================================
    # UPDATE CPU & GPU
    # =====================================
    def update_usage(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        self.ui.PERSENTASE_CPU.setText(f"{cpu_percent:.0f}%")

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.ui.PERSENTASE_GPU.setText(f"{gpu.load * 100:.0f}%")
            else:
                self.ui.PERSENTASE_GPU.setText("N/A")
        except Exception:
            self.ui.PERSENTASE_GPU.setText("N/A")

    # =====================================
    # START KAMERA + MONITOR
    # =====================================
    def start_camera(self):
        if self.camera_thread is not None:
            return

        # Clear lists/counters
        self.ui.TOTAL_JUMLAH.clear()
        
        # (Fix 1: RAM) Memperbaiki Kebocoran RAM saat Merender Log Kecepatan
        layout = self.ui.scrollAreaWidgetContents.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(self.ui.scrollAreaWidgetContents)
            layout.setAlignment(QtCore.Qt.AlignTop)
        else:
            # Hapus widget usang sisa dari klik sebelumnya
            for i in reversed(range(layout.count())): 
                widget_to_remove = layout.itemAt(i).widget()
                if widget_to_remove:
                    layout.removeWidget(widget_to_remove)
                    widget_to_remove.setParent(None)
                    
        # Buat Cukup SATU Label Master penampung riwayat FPS, BUKAN merender Label tak terhingga!
        self.log_inference_label = QtWidgets.QLabel()
        layout.addWidget(self.log_inference_label)
        self.inference_logs = []

        # Menerjemahkan Posisi Layar "Batas Garis" Milik Anda (Qt) ke Posisi Koordinat Kamera Asli 
        batas_y_ui = self.ui.BATAS_GARIS.y() - self.ui.view_kamera.y() + (self.ui.BATAS_GARIS.height() // 2)
        ui_h = self.ui.view_kamera.height()
        ui_w = self.ui.view_kamera.width()
        
        raw_w, raw_h = 1080, 1920
        target_ar = ui_w / ui_h
        raw_ar = raw_w / raw_h
        
        if raw_ar > target_ar:
            raw_target_h = raw_h
            y_offset = 0
        else:
            raw_target_h = int(raw_w / target_ar)
            y_offset = (raw_h - raw_target_h) // 2
            
        ratio_in_ui = batas_y_ui / ui_h
        raw_line_y = int(y_offset + (raw_target_h * ratio_in_ui))

        # Mengirim Y yang sudah sinkron ke AI Thread
        self.camera_thread = CameraThread(camera_index=1, raw_line_y=raw_line_y)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.start()

        # 🔥 START MONITOR CPU & GPU
        self.monitor_timer.start(1000)

        print("Kamera & monitoring (YOLO) dimulai")

    # =====================================
    # UPDATE FRAME
    # =====================================
    def update_frame(self, frame, inference_time_str, current_count):
        # Update object count
        self.ui.TOTAL_JUMLAH.setText(f"{current_count}")
        
        # (Fix 1: RAM) Update inference list menggunakan List Teks Antrean 20 Label
        self.inference_logs.append(f"FPS AI: {inference_time_str}")
        
        # Jika teks melampaui 20 baris, hapus baris paling tua agar kotak log tak tewas
        if len(self.inference_logs) > 20: 
            self.inference_logs.pop(0) 
            
        if hasattr(self, 'log_inference_label'):
            # Menyatukan List Array Log Menjadi Baris Teks Ke Bawah
            self.log_inference_label.setText("\n".join(reversed(self.inference_logs)))

        # === 1) ROTASI/FRAME MASUK MASIH BGR dari thread ===
        # Convert ke RGB untuk Qt
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_w = self.video_label.width()
        target_h = self.video_label.height()
        if target_w <= 0 or target_h <= 0:
            return

        # === 2) TRIM BORDER HITAM (AUTO) ===
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Angka 10 bisa dinaikkan kalau border hitamnya "agak abu"
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # Safety margin biar tidak kepotong terlalu mepet
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            frame = frame[y1:y2, x1:x2]

        # === 3) CENTER CROP KE RASIO QLabel ===
        h, w, _ = frame.shape
        target_ar = target_w / target_h
        frame_ar = w / h

        if frame_ar > target_ar:
            new_w = int(h * target_ar)
            x1 = (w - new_w) // 2
            frame = frame[:, x1:x1 + new_w]
        else:
            new_h = int(w / target_ar)
            y1 = (h - new_h) // 2
            frame = frame[y1:y1 + new_h, :]

        # === 4) RESIZE FULL (TANPA SISA KOSONG) ===
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # === 5) TAMPILKAN KE QLabel ===
        h2, w2, ch2 = frame.shape
        bytes_per_line = ch2 * w2
        qt_image = QtGui.QImage(frame.data, w2, h2, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))


    # =====================================
    # STOP KAMERA + MONITOR
    # =====================================
    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        # ⛔ STOP MONITOR
        if self.monitor_timer.isActive():
            self.monitor_timer.stop()

        self.video_label.clear()
        self.ui.PERSENTASE_CPU.setText("0%")
        self.ui.PERSENTASE_GPU.setText("0%")

        print("Kamera & monitoring dihentikan")

    # =====================================
    # CLOSE EVENT
    # =====================================
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

    # =====================================
    # TENTANG EVENT
    # =====================================      
    def eventFilter(self, obj, event):
        if obj == self.ui.label_TENTANG:
            if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                self.show_about_popup()
                return True
        return super().eventFilter(obj, event)
    def show_about_popup(self):
        QtWidgets.QMessageBox.information(
            self,
            "Tentang Sistem",
            (
                "Nama Project : Kamera Monitoring\n"
                "Deskripsi    : Preview kamera + monitoring CPU/GPU\n"
                "Dibuat oleh  : (isi nama Anda)\n"
                "Tahun        : 2025\n"
                "Versi        : 1.0.0\n"
                "Catatan      : Klik MULAI untuk menjalankan kamera & monitoring\n"
            )
        )

# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
