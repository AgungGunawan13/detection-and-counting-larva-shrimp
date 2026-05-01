import sys
from ultralytics import YOLO
import cv2
import psutil
import GPUtil
import time
from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_Dialog
import queue


# =========================================
# THREAD CAPTURE (PRODUCER)
# =========================================
class CaptureThread(QtCore.QThread):
    def __init__(self, camera_index=1):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.frame_queue = queue.Queue()
        self.is_finished = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)

        if not cap.isOpened():
            self.is_finished = True
            return

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                self.frame_queue.put(buffer)

        cap.release()
        self.is_finished = True

    def stop(self):
        self.running = False


# =========================================
# THREAD KAMERA (CONSUMER / AI)
# =========================================
class CameraThread(QtCore.QThread):
    # Emit frame, inference time string, updated count, q_size
    frame_ready = QtCore.pyqtSignal(object, str, int, int)
    error_signal = QtCore.pyqtSignal(str)
    processing_finished = QtCore.pyqtSignal()

    def __init__(self, capture_thread, ratio_in_ui=0.5, ui_w=1080, ui_h=1920):
        super().__init__()
        self.capture_thread = capture_thread
        self.ratio_in_ui = ratio_in_ui
        self.ui_w = ui_w
        self.ui_h = ui_h
        self.running = False
        self.draining = False
        
        from pathlib import Path
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        
        # Load local YOLO model
        self.weights_path = str(self.root_dir / "models" / "trained" / "best.pt")
        self.model = YOLO(self.weights_path)
        
        self.line_y = None
        self.count = 0
        
        self.track_history = {}
        self.track_paths = {}
        self.counted_ids = set()
        self.frame_count = 0
        self.offset = 20
        self.arah = "TOP_BOTTOM"
        self.conf_threshold = 0.15
        self.id_expiry_frames = 90
        self.total_inference_time = 0.0
        
        self.tracker_path = str(self.root_dir / "src" / "scripts" / "custom_tracker.yaml")

    def run(self):
        self.running = True

        while self.running or self.draining:
            if not self.capture_thread.frame_queue.empty():
                buffer = self.capture_thread.frame_queue.get()
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                
                q_size = self.capture_thread.frame_queue.qsize()

                if self.line_y is None:
                    h, w, _ = frame.shape
                    target_ar = self.ui_w / self.ui_h if self.ui_h > 0 else 1
                    frame_ar = w / h if h > 0 else 1
                    
                    if frame_ar < target_ar:
                        self.line_y = int(h * self.ratio_in_ui)
                    else:
                        scale = self.ui_w / w
                        new_h = h * scale
                        pad_y = (self.ui_h - new_h) / 2
                        batas_y_ui = self.ui_h * self.ratio_in_ui
                        y_scaled = batas_y_ui - pad_y
                        self.line_y = int(y_scaled / scale)
                        
                    self.line_y = max(0, min(h - 1, self.line_y))

                # Start timing inference
                start_time = time.time()
                self.frame_count += 1
                
                # Predict with YOLO GPU & Tracker Config parameter anti bayangan semu
                results = self.model.track(
                    source=frame,
                    conf=self.conf_threshold,
                    imgsz=640,
                    device=0,
                    tracker=self.tracker_path,
                    persist=True,
                    verbose=False
                )
                
                # Calculate inference time
                inference_time = (time.time() - start_time) * 1000 # in ms
                self.total_inference_time += inference_time
                inf_time_str = f"{inference_time:.2f} ms"

                frame = results[0].plot()

                garis_atas = self.line_y - self.offset
                garis_bawah = self.line_y + self.offset

                # Gambar garis zona
                cv2.line(frame, (0, garis_atas), (frame.shape[1], garis_atas), (255, 200, 0), 1)
                cv2.line(frame, (0, garis_bawah), (frame.shape[1], garis_bawah), (255, 200, 0), 1)
                cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0, 0, 255), 3)

                # Process tracked objects
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    boxes     = results[0].boxes.xyxy.cpu().tolist()
                    
                    for track_id, box in zip(track_ids, boxes):
                        x1, y1, x2, y2 = box
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        cv2.circle(frame, (cx, cy), 6, (255, 255, 0), -1)

                        if track_id not in self.track_paths:
                            self.track_paths[track_id] = []
                        self.track_paths[track_id].append((cx, cy))
                        if len(self.track_paths[track_id]) > 40:
                            self.track_paths[track_id].pop(0)
                        for i in range(1, len(self.track_paths[track_id])):
                            cv2.line(frame,
                                     self.track_paths[track_id][i-1],
                                     self.track_paths[track_id][i], (0, 165, 255), 2)
                        
                        # Check for line crossing (Zone-crossing logic)
                        if track_id not in self.counted_ids:
                            info = self.track_history.get(track_id, {'zone': None, 'last_seen': self.frame_count})
                            info['last_seen'] = self.frame_count
                            
                            if self.arah == "TOP_BOTTOM":
                                if cy < garis_atas:
                                    info['zone'] = 'ATAS'
                                elif cy > garis_bawah and info['zone'] == 'ATAS':
                                    self.count += 1
                                    self.counted_ids.add(track_id)
                                    info['zone'] = 'BAWAH'
                            elif self.arah == "BOTTOM_TOP":
                                if cy > garis_bawah:
                                    info['zone'] = 'BAWAH'
                                elif cy < garis_atas and info['zone'] == 'BAWAH':
                                    self.count += 1
                                    self.counted_ids.add(track_id)
                                    info['zone'] = 'ATAS'
                                    
                            self.track_history[track_id] = info

                # Bersihkan ID kedaluwarsa
                expired = [
                    tid for tid, info in self.track_history.items()
                    if self.frame_count - info.get('last_seen', 0) > self.id_expiry_frames
                    and tid not in self.counted_ids
                ]
                for tid in expired:
                    self.track_history.pop(tid, None)
                    self.track_paths.pop(tid, None)

                self.frame_ready.emit(frame, inf_time_str, self.count, q_size)
            else:
                if self.draining and self.capture_thread.is_finished:
                    break
                else:
                    time.sleep(0.01)

        self.processing_finished.emit()

    def start_draining(self):
        self.draining = True
        self.running = False

    def stop_immediate(self):
        self.running = False
        self.draining = False


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
        self.capture_thread = None
        self.cpu_history = []
        self.gpu_history = []
        self.start_time_uji = None

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

        # PENGATURAN CONFIDENCE & ARAH HARDCODED DI CAMERA THREAD
        
        # Button
        self.ui.MULAI.clicked.connect(self.start_camera)
        self.ui.SELESAI.clicked.connect(self.stop_camera)
        self.ui.CLOSE.clicked.connect(self.close)

    # =====================================
    # UPDATE PENGGUNAAN
    # =====================================
    def handle_camera_error(self, error_msg):
        self.stop_camera()
        QtWidgets.QMessageBox.critical(self, "Error Kamera", error_msg)

    # =====================================
    # UPDATE CPU & GPU
    # =====================================
    def update_usage(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        self.ui.PERSENTASE_CPU.setText(f"{cpu_percent:.0f}%")
        self.cpu_history.append(cpu_percent)

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load = gpu.load * 100
                self.ui.PERSENTASE_GPU.setText(f"{gpu_load:.0f}%")
                self.gpu_history.append(gpu_load)
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
        self.cpu_history.clear()
        self.gpu_history.clear()
        self.start_time_uji = time.time()
        
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

        ui_h = self.ui.view_kamera.height()
        ui_w = self.ui.view_kamera.width()
        
        # Menggunakan rasio garis statis 0.6 mirip dengan test_video
        ratio_in_ui = 0.6

        # Initialize and start Producer
        self.capture_thread = CaptureThread(camera_index=1)
        self.capture_thread.start()

        # Mengirim parameter UI ke AI Thread agar menghitung Y secara dinamis
        self.camera_thread = CameraThread(capture_thread=self.capture_thread, ratio_in_ui=ratio_in_ui, ui_w=ui_w, ui_h=ui_h)

        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.error_signal.connect(self.handle_camera_error)
        self.camera_thread.processing_finished.connect(self.generate_report_and_reset)
        self.camera_thread.start()

        # Reset button status in case it was used before
        self.ui.SELESAI.setEnabled(True)
        self.ui.SELESAI.setText("SELESAI")

        # 🔥 START MONITOR CPU & GPU
        self.monitor_timer.start(1000)

        print("Kamera & monitoring (YOLO) dimulai")

    # =====================================
    # UPDATE FRAME
    # =====================================
    def update_frame(self, frame, inference_time_str, current_count, q_size):
        # Update object count
        self.ui.TOTAL_JUMLAH.setText(f"{current_count}")
        
        # Update status tombol jika sedang draining
        if self.camera_thread and self.camera_thread.draining:
            self.ui.SELESAI.setText(f"Memproses Sisa ({q_size} frame)...")
            
        # ========================================
        # PEREKAMAN VIDEO DENGAN OVERLAY PARAMETER
        # ========================================
        import numpy as np
        if not hasattr(self, 'video_writer') or self.video_writer is None:
            from pathlib import Path
            import time
            report_dir = Path(__file__).resolve().parent / "laporan_pengujian"
            report_dir.mkdir(parents=True, exist_ok=True)
            self.video_filename = report_dir / f"Rekaman_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h_vid, w_vid, _ = frame.shape
            self.video_writer = cv2.VideoWriter(str(self.video_filename), fourcc, 30.0, (w_vid, h_vid))

        # Buat salinan frame khusus untuk direkam (agar tidak mengganggu render UI utama)
        rec_frame = frame.copy()
        
        # Gambar background hitam semi transparan untuk teks
        cv2.rectangle(rec_frame, (10, 10), (480, 160), (0, 0, 0), -1)
        cv2.addWeighted(rec_frame, 0.5, frame, 0.5, 0, rec_frame)
        
        # Tambahkan teks overlay
        cpu_text = self.ui.PERSENTASE_CPU.text()
        gpu_text = self.ui.PERSENTASE_GPU.text()
        cv2.putText(rec_frame, f"Total Benur: {current_count}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(rec_frame, f"inferensi : {inference_time_str}", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(rec_frame, f"CPU: {cpu_text}  GPU: {gpu_text}", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Tulis ke video
        self.video_writer.write(rec_frame)
        
        # Menghitung estimasi FPS dari waktu inferensi
        try:
            ms_val = float(inference_time_str.split()[0])
            fps_val = int(1000 / ms_val) if ms_val > 0 else 0
            log_text = f"Sistem: {fps_val} FPS | Inferensi: {inference_time_str}"
        except:
            log_text = f"Sistem: {inference_time_str}"
            
        # (Fix 1: RAM) Update inference list menggunakan List Teks Antrean 20 Label
        self.inference_logs.append(log_text)
        
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

        # (Fitur Trim Border ditiadakan karena dapat menggeser koordinat garis perhitungan)

        # === 3) FIT TO WINDOW (MAINTAIN ASPECT RATIO) ===
        import numpy as np
        h, w, _ = frame.shape
        target_ar = target_w / target_h
        frame_ar = w / h

        if frame_ar < target_ar:
            new_h = target_h
            new_w = int(w * (target_h / h))
        else:
            new_w = target_w
            new_h = int(h * (target_w / w))

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
        frame = canvas

        # === 5) TAMPILKAN KE QLabel ===
        h2, w2, ch2 = frame.shape
        bytes_per_line = ch2 * w2
        qt_image = QtGui.QImage(frame.data, w2, h2, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))


    # =====================================
    # STOP KAMERA + MONITOR + EXPORT
    # =====================================
    def stop_camera(self):
        if self.camera_thread and (self.camera_thread.running or self.camera_thread.draining):
            if self.capture_thread and self.capture_thread.running:
                self.capture_thread.stop()
            
            self.camera_thread.start_draining()
            
            self.ui.SELESAI.setEnabled(False)
            self.ui.SELESAI.setText("Menghentikan Kamera...")
            
            # ⛔ STOP MONITOR
            if self.monitor_timer.isActive():
                self.monitor_timer.stop()

    def generate_report_and_reset(self):
        # Stop video writer
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.camera_thread:
            # Ambil data sebelum thread dihancurkan
            total_benur = self.camera_thread.count
            total_frames = self.camera_thread.frame_count
            total_inf_time = self.camera_thread.total_inference_time
            arah = self.camera_thread.arah
            conf = self.camera_thread.conf_threshold
            
            self.camera_thread.stop_immediate()
            self.camera_thread = None
            self.capture_thread = None
            
            # Hitung rata-rata
            avg_inf = (total_inf_time / total_frames) if total_frames > 0 else 0
            avg_fps = (1000 / avg_inf) if avg_inf > 0 else 0
            
            avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
            avg_gpu = sum(self.gpu_history) / len(self.gpu_history) if self.gpu_history else 0
            
            durasi_detik = time.time() - self.start_time_uji if self.start_time_uji else 0
            menit, detik = divmod(int(durasi_detik), 60)
            
            # Buat teks laporan
            waktu_skrg = time.strftime("%Y-%m-%d %H:%M:%S")
            laporan = (
                f"====================================\n"
                f"   LAPORAN PENGUJIAN REAL-TIME\n"
                f"====================================\n"
                f"Waktu Pengujian : {waktu_skrg}\n"
                f"Durasi          : {menit} menit {detik} detik\n"
                f"Konfigurasi     : Arah={arah}, Conf={conf:.2f}\n"
                f"------------------------------------\n"
                f"HASIL DETEKSI\n"
                f"Total Benur     : {total_benur} ekor\n"
                f"Total Frame     : {total_frames} frame\n"
                f"------------------------------------\n"
                f"PERFORMA SISTEM\n"
                f"Rata-rata CPU   : {avg_cpu:.1f} %\n"
                f"Rata-rata GPU   : {avg_gpu:.1f} %\n"
                f"Inference Time  : {avg_inf:.2f} ms/frame\n"
                f"Rata-rata FPS   : {avg_fps:.2f} FPS\n"
                f"====================================\n"
            )
            
            # Simpan ke folder laporan_pengujian
            from pathlib import Path
            report_dir = Path(__file__).resolve().parent / "laporan_pengujian"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            filename = report_dir / f"Laporan_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w") as f:
                f.write(laporan)
                
            QtWidgets.QMessageBox.information(self, "Laporan Disimpan", f"Laporan pengujian telah disimpan ke:\n{filename}")

        self.video_label.clear()
        self.ui.PERSENTASE_CPU.setText("0%")
        self.ui.PERSENTASE_GPU.setText("0%")
        self.ui.SELESAI.setEnabled(True)
        self.ui.SELESAI.setText("SELESAI")

        print("Kamera & monitoring dihentikan sepenuhnya")

    # =====================================
    # CLOSE EVENT
    # =====================================
    def closeEvent(self, event):
        if self.capture_thread:
            self.capture_thread.stop()
        if self.camera_thread:
            self.camera_thread.stop_immediate()
            self.camera_thread.wait()
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
                "Nama Project : Sistem deteksi dan perhitungan secara real-time benur udang windu\n"
                "Deskripsi    : sistem ini berguna untuk menyelesaikan skripsi\n"
                "Dibuat oleh  : Agung Gunawan\n"
                "Tahun        : 2026\n"
                "Versi        : 1.1.0\n"
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
