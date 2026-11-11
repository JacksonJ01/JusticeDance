"""Justice Dance PyQt6 desktop game.

The reference video fills the stage. The live camera is composited as a
picture-in-picture feed and its pose rows are saved per player as Parquet.
When a MediaPipe Tasks .task model is selected, the same landmark/feature
conventions as AIHuMo are used for extraction and scoring.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import time

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.justice_analysis import build_features, score, write_player_parquet  # noqa: E402


class PoseCollector:
    def __init__(self, model_path: str):
        self.landmarker = None
        self.model_path = model_path
        if model_path:
            try:
                import mediapipe as mp
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=.5,
                    min_pose_presence_confidence=.5,
                    min_tracking_confidence=.5,
                )
                self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
                self.mp = mp
            except Exception as exc:
                raise RuntimeError(f"Could not initialize MediaPipe: {exc}") from exc

    def row(self, frame_bgr: np.ndarray, frame_id: int, timestamp_s: float, player_id: str) -> dict:
        row = {"player_id": player_id, "frame_id": frame_id, "timestamp_s": timestamp_s, "pose_present": 0}
        if self.landmarker is None:
            return row
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, int(timestamp_s * 1000))
        pose = result.pose_landmarks[0] if result.pose_landmarks else None
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        row["pose_present"] = int(pose is not None)
        for i in range(33):
            point = pose[i] if pose else None
            world_point = world[i] if world else None
            for axis in "xyz":
                row[f"lm{i}_{axis}"] = float(getattr(point, axis)) if point else np.nan
                row[f"world{i}_{axis}"] = float(getattr(world_point, axis)) if world_point else np.nan
            row[f"lm{i}_visibility"] = float(getattr(point, "visibility", np.nan)) if point else np.nan
            row[f"lm{i}_presence"] = float(getattr(point, "presence", np.nan)) if point else np.nan
        return row

    def close(self):
        if self.landmarker:
            self.landmarker.close()


class JusticeDanceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Justice Dance — AIHuMo Dance Lab")
        self.resize(1280, 820)
        self.reference = None
        self.camera = None
        self.reference_path = ""
        self.collector = None
        self.rows: list[dict] = []
        self.frame_id = 0
        self.started_at = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.stage = QLabel("Choose a reference video to begin")
        self.stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage.setMinimumSize(900, 560)
        self.stage.setStyleSheet("background:#14192b;color:#f7f1e8;font-size:22px")
        self.status = QLabel("Reference video: none · Camera: stopped · Features: not recording")
        self.reference_edit = QLineEdit()
        self.model_edit = QLineEdit()
        candidates = [
            Path.home() / "Downloads" / "AIHuMo_refactored_v0_2" / "full_model" / "pose_landmarker_full.task",
            Path.home() / "Downloads" / "Humo_app" / "full_model" / "pose_landmarker_full.task",
        ]
        for shared_model in candidates:
            if shared_model.exists():
                self.model_edit.setText(str(shared_model))
                break
        self.player_spin = QSpinBox(); self.player_spin.setRange(1, 2); self.player_spin.setValue(1)
        browse_ref = QPushButton("Choose reference video"); browse_ref.clicked.connect(self.choose_reference)
        browse_model = QPushButton("Choose MediaPipe model"); browse_model.clicked.connect(self.choose_model)
        camera_btn = QPushButton("Start camera"); camera_btn.clicked.connect(self.toggle_camera); self.camera_btn = camera_btn
        start_btn = QPushButton("Start dance"); start_btn.clicked.connect(self.start_dance); self.start_btn = start_btn
        stop_btn = QPushButton("Stop & save Parquet"); stop_btn.clicked.connect(self.stop_dance)
        controls = QGridLayout(); controls.addWidget(QLabel("Reference"),0,0); controls.addWidget(self.reference_edit,0,1); controls.addWidget(browse_ref,0,2)
        controls.addWidget(QLabel("MediaPipe .task"),1,0); controls.addWidget(self.model_edit,1,1); controls.addWidget(browse_model,1,2)
        controls.addWidget(QLabel("Players"),2,0); controls.addWidget(self.player_spin,2,1); controls.addWidget(camera_btn,2,2); controls.addWidget(start_btn,2,3); controls.addWidget(stop_btn,2,4)
        layout=QVBoxLayout(); layout.addLayout(controls); layout.addWidget(self.stage,1); layout.addWidget(self.status)
        root=QWidget(); root.setLayout(layout); self.setCentralWidget(root)

    def choose_reference(self):
        path,_=QFileDialog.getOpenFileName(self,"Choose reference dance video","","Video (*.mp4 *.mov *.avi *.mkv *.webm)")
        if path:
            self.reference_path=path; self.reference_edit.setText(path); self.reference=cv2.VideoCapture(path); self.status.setText(f"Reference video: {Path(path).name} · Camera: stopped")

    def choose_model(self):
        path,_=QFileDialog.getOpenFileName(self,"Choose MediaPipe pose model",str(ROOT),"MediaPipe model (*.task)")
        if path: self.model_edit.setText(path)

    def toggle_camera(self):
        if self.camera:
            self.camera.release(); self.camera=None; self.camera_btn.setText("Start camera"); return
        self.camera=cv2.VideoCapture(0); self.camera_btn.setText("Stop camera")
        if not self.camera.isOpened(): self.camera=None; QMessageBox.warning(self,"Camera unavailable","Justice Dance could not open camera 0.")

    def start_dance(self):
        if not self.reference: QMessageBox.warning(self,"Reference required","Choose a downloaded reference video first."); return
        try:
            self.collector=PoseCollector(self.model_edit.text().strip())
        except RuntimeError as exc: QMessageBox.critical(self,"MediaPipe error",str(exc)); return
        self.rows=[]; self.frame_id=0; self.started_at=time(); self.start_btn.setEnabled(False); self.timer.start(33); self.status.setText("Dancing: reference large · live camera picture-in-picture · collecting features")

    def stop_dance(self):
        if not self.started_at: return
        self.timer.stop(); self.start_btn.setEnabled(True)
        if self.collector:
            self.collector.close(); self.collector=None
        if self.rows:
            player_id=f"player_{self.player_spin.value()}"; raw=pd.DataFrame(self.rows); out=ROOT/"data"/"justice_sessions"/player_id
            out.mkdir(parents=True,exist_ok=True); raw_path=write_player_parquet(raw,out/f"{player_id}_landmarks.parquet",player_id)
            try: feature_path=write_player_parquet(build_features(raw),out/f"{player_id}_features.parquet",player_id)
            except Exception: feature_path=None
            self.status.setText(f"Saved {raw_path.name}" + (f" and {feature_path.name}" if feature_path else "") + " · each player has a separate dataset")
        self.started_at=0.0

    def next_frame(self):
        if not self.reference: return
        ok, ref=self.reference.read()
        if not ok: self.reference.set(cv2.CAP_PROP_POS_FRAMES,0); ok,ref=self.reference.read()
        if not ok: return
        if self.camera and self.camera.isOpened():
            cam_ok, cam=self.camera.read()
            if cam_ok:
                cam=cv2.resize(cam,(max(1,int(ref.shape[1]*.28)),max(1,int(ref.shape[0]*.28))))
                ref[20:20+cam.shape[0],20:20+cam.shape[1]]=cam
                if self.started_at and self.collector: self.rows.append(self.collector.row(cam,self.frame_id,time()-self.started_at,f"player_{self.player_spin.value()}")); self.frame_id+=1
        rgb=cv2.cvtColor(ref,cv2.COLOR_BGR2RGB); h,w,ch=rgb.shape; image=QImage(rgb.data,w,h,ch*w,QImage.Format.Format_RGB888); self.stage.setPixmap(QPixmap.fromImage(image).scaled(self.stage.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self,event):
        self.stop_dance()
        if self.camera: self.camera.release()
        if self.reference: self.reference.release()
        event.accept()


if __name__ == "__main__":
    app=QApplication(sys.argv); window=JusticeDanceWindow(); window.show(); sys.exit(app.exec())
