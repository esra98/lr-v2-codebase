#!/usr/bin/env python3
"""Canlı demo: webcam'den bir kayıt (ya da bir video dosyası) -> dudak okuma
modeli (yalnız görüntü) ve Whisper (yalnız ses), yan yana metin ve WER/CER.

    .venv/bin/python live-demo/dudak_okuma.py [--run <model klasörü>] [--seconds 10]

Başlat ile kayıt başlar, Durdur ile (ya da 10/20/30 sn sınırında) biter. Kayıtlar
~/Videos/dudak_okuma/<tarih>/ altına yazılır (kayit.mp4, agiz.mp4, ses.wav,
strip.png, sonuc.json) ve uygulamadaki geçmiş listesinden açılıp silinebilir.
"""

import argparse
import datetime
import fcntl
import gc
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from fractions import Fraction

import cv2  # noqa: E402  (before Qt: cv2 sets a Qt5 plugin path we must undo)
import numpy as np

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402

DEFAULT_RUN = os.path.expanduser("~/Desktop/vast-run/models/tr-vsr-model")
OUT_ROOT = os.path.expanduser("~/Videos/dudak_okuma")


def keep_framerate(device="/dev/video0"):
    """Turn off V4L2 exposure_auto_priority: in dim light the camera would
    otherwise lengthen the exposure and drop from 30 to 10 fps."""
    try:
        fd = os.open(device, os.O_RDWR)
        try:
            fcntl.ioctl(fd, 0xC008561C, struct.pack("Ii", 0x009A0903, 0))  # VIDIOC_S_CTRL
        finally:
            os.close(fd)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Models (run in a worker thread)
# --------------------------------------------------------------------------
class Worker(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    step = QtCore.pyqtSignal(int)
    ready = QtCore.pyqtSignal()
    done = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, run, beam, whisper):
        super().__init__()
        self.run, self.beam, self.whisper_name = run, beam, whisper

    @QtCore.pyqtSlot()
    def load(self):
        try:
            import torch
            from faster_whisper import WhisperModel

            from avsr.inference import load_trained

            self.status.emit("Dudak okuma modeli yükleniyor…")
            module, _, _ = load_trained(self.run, average=1)
            self.module = module.cuda().eval()
            self.status.emit(f"Whisper {self.whisper_name} yükleniyor…")
            self.whisper = WhisperModel(self.whisper_name, device="cuda", compute_type="int8_float16")
            self.torch = torch
            self.ready.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Model yüklenemedi: {e}")

    @QtCore.pyqtSlot(str, str, str)
    def process(self, video_path, audio_path, out_dir):
        try:
            self.done.emit(self._process(video_path, audio_path, out_dir))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Hata: {e}")

    @QtCore.pyqtSlot(str, object, object)
    def process_recording(self, out_dir, frames, fps):
        """Encode the webcam take (here, not in the GUI thread), then process it."""
        try:
            self.step.emit(0)
            wav = os.path.join(out_dir, "ses.wav")
            video = os.path.join(out_dir, "kayit.mp4")
            h, w = frames[0].shape[:2]
            # Frames are piped to ffmpeg (no big array, no GIL-bound encoding in Python).
            # The system ffmpeg has no libx264: NVENC on the GPU, else OpenH264.
            for codec in (["h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18"],
                          ["libopenh264", "-b:v", "12M"]):
                enc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-i", wav, "-c:v", *codec,
                     "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", video],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                try:
                    for f in frames:
                        enc.stdin.write(f.tobytes())
                    enc.stdin.close()
                except BrokenPipeError:
                    pass
                if enc.wait() == 0:
                    break
            else:
                raise RuntimeError("video yazılamadı (ffmpeg)")
            del frames
            self.done.emit(self._process(video, wav, out_dir))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Hata: {e}")

    def _process(self, video_path, audio_path, out_dir):
        from avsr.inference import mouth_crops, transcribe
        from avsr.preprocess import video_io
        from avsr.text.normalize import normalize_text
        from avsr.utils.metrics import edit_distance

        t0 = time.time()
        self.step.emit(1)
        crops, found = mouth_crops(video_path)
        video_io.write_video(os.path.join(out_dir, "agiz.mp4"), crops)

        self.step.emit(2)
        with self.torch.no_grad():
            parts = transcribe(self.module, crops, beam_size=self.beam)
        lr = " ".join(p[2] for p in parts).strip()
        t_lr = time.time() - t0

        self.step.emit(3)
        t1 = time.time()
        segs, _ = self.whisper.transcribe(audio_path, language="tr", beam_size=5, vad_filter=True,
                                          condition_on_previous_text=False)
        # Whisper invents text over silence ("izlediğiniz için teşekkürler"); drop such segments.
        whisper_raw = " ".join(s.text.strip() for s in segs
                               if s.no_speech_prob < 0.5 and s.avg_logprob > -1.0).strip()
        whisper = normalize_text(whisper_raw)
        t_wh = time.time() - t1
        gc.collect()

        ref, hyp = whisper.split(), lr.split()
        wer = edit_distance(ref, hyp) / len(ref) if ref else None
        rc, hc = whisper.replace(" ", ""), lr.replace(" ", "")
        cer = edit_distance(list(rc), list(hc)) / len(rc) if rc else None

        idx = np.linspace(0, len(crops) - 1, num=min(12, len(crops))).astype(int)
        strip = np.ascontiguousarray(np.hstack([crops[i] for i in idx]))
        cv2.imwrite(os.path.join(out_dir, "strip.png"), strip[:, :, ::-1])

        result = {"video": video_path, "frames": len(crops), "seconds": round(len(crops) / 25, 2),
                  "face_found": round(float(found), 3), "lip_reading": lr, "whisper": whisper,
                  "whisper_raw": whisper_raw, "wer": wer, "cer": cer,
                  "time_lr_s": round(t_lr, 1), "time_whisper_s": round(t_wh, 1)}
        with open(os.path.join(out_dir, "sonuc.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        result["strip"] = strip
        result["out_dir"] = out_dir
        return result


# --------------------------------------------------------------------------
# Camera (own thread)
# --------------------------------------------------------------------------
class Camera(QtCore.QThread):
    frame = QtCore.pyqtSignal(QtGui.QImage, float, float)
    error = QtCore.pyqtSignal(str)

    def __init__(self, index):
        super().__init__()
        self.index, self.running = index, True
        self.recording, self.frames = False, []

    def run(self):
        keep_framerate(f"/dev/video{self.index}")
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            self.error.emit("Kamera açılamadı.")
            return
        keep_framerate(f"/dev/video{self.index}")
        times = []
        while self.running:
            ok, bgr = cap.read()
            if not ok:
                continue
            now = time.time()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.recording:
                self.frames.append((now, rgb))
            times = [t for t in times if now - t < 1.0] + [now]
            view = np.ascontiguousarray(rgb[:, ::-1])  # mirror for the preview only
            h, w, _ = view.shape
            img = QtGui.QImage(view.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
            self.frame.emit(img, float(len(times)), float(bgr.mean()))
        cap.release()


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------
BG, CARD, LINE, TEXT, MUTED = "#0f1115", "#171a21", "#262a33", "#e6e8ee", "#8b93a7"
RED, BLUE, GREEN, AMBER = "#e5484d", "#7aa7ff", "#3dd68c", "#f5b84a"

STYLE = f"""
QWidget {{ background:{BG}; color:{TEXT}; font-family:'Noto Sans','Cantarell',sans-serif; font-size:14px; }}
QLabel {{ background:transparent; }}
QFrame#card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; }}
QFrame#card QWidget {{ background:transparent; }}
QLabel#muted {{ color:{MUTED}; font-size:12px; }}
QLabel#h {{ font-size:12px; font-weight:600; letter-spacing:1px; color:{MUTED}; }}
QPushButton {{ background:#232733; border:1px solid #2f3442; border-radius:10px; padding:9px 16px; }}
QPushButton:hover {{ background:#2b3040; }}
QPushButton:disabled {{ color:#5b6273; background:#1b1e26; }}
QPushButton#start {{ background:{RED}; border:none; color:white; font-size:17px; font-weight:700;
                     padding:13px 30px; border-radius:24px; }}
QPushButton#start:hover {{ background:#f2555a; }}
QPushButton#start[mode="stop"] {{ background:#f2f3f5; color:#111; }}
QPushButton#start:disabled {{ background:#3a2427; color:#9a7476; }}
QPushButton#seg {{ border-radius:8px; padding:7px 12px; }}
QPushButton#seg:checked {{ background:#3b4252; border-color:#6b7389; }}
QPushButton#danger {{ color:#ff8a8d; }}
QListWidget {{ background:transparent; border:none; outline:0; }}
QListWidget::item {{ padding:8px 6px; border-radius:8px; color:{TEXT}; }}
QListWidget::item:hover {{ background:#1f232d; }}
QListWidget::item:selected {{ background:#2a3040; }}
QScrollBar:vertical {{ background:transparent; width:8px; }}
QScrollBar::handle:vertical {{ background:#333948; border-radius:4px; min-height:30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; }}
"""


def card():
    f = QtWidgets.QFrame(objectName="card")
    lay = QtWidgets.QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(8)
    return f, lay


def align(ref, hyp):
    """Word alignment: tags for ref words (ok/sub/del) and hyp words (ok/sub/ins)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    rt, ht, i, j = [None] * n, [None] * m, n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            tag = "ok" if ref[i - 1] == hyp[j - 1] else "sub"
            rt[i - 1] = ht[j - 1] = tag
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            rt[i - 1] = "del"
            i -= 1
        else:
            ht[j - 1] = "ins"
            j -= 1
    return rt, ht


def words_html(words, tags, wrong_color):
    out = []
    for w, t in zip(words, tags):
        if t == "ok":
            out.append(f"<span style='color:{TEXT}'>{w}</span>")
        else:
            out.append(f"<span style='color:{wrong_color}'>{w}</span>")
    return " ".join(out)


class CameraView(QtWidgets.QWidget):
    """Live picture, cover-cropped into a rounded frame, with the state drawn on top."""

    def __init__(self):
        super().__init__()
        self.img, self.fps, self.bright = None, 0.0, 0.0
        self.state, self.count, self.rec_start, self.rec_limit, self.note = "idle", 0, 0.0, 10.0, ""
        self.setMinimumSize(640, 400)

    def set_frame(self, img, fps, bright):
        self.img, self.fps, self.bright = img, fps, bright
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        r = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, 16, 16)
        p.setClipPath(path)
        p.fillRect(r, QtGui.QColor("#000"))
        if self.img is None:
            p.setPen(QtGui.QColor(MUTED))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Kamera açılıyor…")
            return
        iw, ih = self.img.width(), self.img.height()
        s = max(r.width() / iw, r.height() / ih)
        w, h = iw * s, ih * s
        p.drawImage(QtCore.QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h), self.img)
        cx, cy = r.center().x(), r.center().y()

        if self.state == "idle":
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            ow, oh = r.height() * 0.42, r.height() * 0.6
            p.drawEllipse(QtCore.QRectF(cx - ow / 2, cy - oh / 2 - r.height() * 0.03, ow, oh))
            self.pill(p, r.width() / 2, r.height() - 34, "Yüzünüzü çerçeveye alın", center=True)
        elif self.state == "countdown":
            p.fillRect(r, QtGui.QColor(0, 0, 0, 120))
            p.setPen(QtGui.QColor("white"))
            f = QtGui.QFont("Noto Sans", 120, QtGui.QFont.Weight.Bold)
            p.setFont(f)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(self.count))
            self.pill(p, r.width() / 2, r.height() - 34, "Hazırlanın…", center=True)
        elif self.state == "recording":
            el = time.time() - self.rec_start
            p.setPen(QtGui.QPen(QtGui.QColor(RED), 6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(3, 3, -3, -3), 14, 14)
            self.pill(p, 20, 20, f"●  KAYIT  {el:4.1f} / {self.rec_limit:g} sn", color=RED)
            frac = min(1.0, el / self.rec_limit)
            p.fillRect(QtCore.QRectF(0, r.height() - 8, r.width(), 8), QtGui.QColor(0, 0, 0, 140))
            p.fillRect(QtCore.QRectF(0, r.height() - 8, r.width() * frac, 8), QtGui.QColor(RED))
        elif self.state == "busy":
            p.fillRect(r, QtGui.QColor(0, 0, 0, 150))
            p.setPen(QtGui.QColor("white"))
            p.setFont(QtGui.QFont("Noto Sans", 20, QtGui.QFont.Weight.DemiBold))
            pen = QtGui.QPen(QtGui.QColor(BLUE), 6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            arc = QtCore.QRectF(cx - 34, cy - 70, 68, 68)
            p.drawArc(arc, int(-time.time() * 360 * 16) % (360 * 16), 100 * 16)
            p.setPen(QtGui.QColor("white"))
            p.drawText(QtCore.QRectF(r.left(), cy + 10, r.width(), 40), Qt.AlignmentFlag.AlignCenter, self.note)
            p.setFont(QtGui.QFont("Noto Sans", 12))
            p.setPen(QtGui.QColor(MUTED))
            p.drawText(QtCore.QRectF(r.left(), cy + 50, r.width(), 30), Qt.AlignmentFlag.AlignCenter,
                       "model çalışıyor, birkaç saniye…")

        ok = self.fps >= 20 and self.bright >= 50
        text = f"{self.fps:.0f} fps" + ("" if ok else ("  ·  ışık az" if self.fps >= 20 or self.bright < 50 else ""))
        self.pill(p, r.width() - 20, 20, text, color=GREEN if ok else AMBER, right=True)

    def pill(self, p, x, y, text, color=None, center=False, right=False):
        p.setFont(QtGui.QFont("Noto Sans", 11, QtGui.QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 24, fm.height() + 12
        if center:
            x -= w / 2
        if right:
            x -= w
        rect = QtCore.QRectF(x, y, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 160))
        p.drawRoundedRect(rect, h / 2, h / 2)
        p.setPen(QtGui.QColor(color or "white"))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class Stat(QtWidgets.QFrame):
    def __init__(self, label):
        super().__init__(objectName="card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(0)
        self.value = QtWidgets.QLabel("—")
        self.value.setStyleSheet("font-size:28px; font-weight:700;")
        cap = QtWidgets.QLabel(label, objectName="muted")
        lay.addWidget(self.value)
        lay.addWidget(cap)


STEPS = ["Video kaydediliyor", "Yüz ve ağız bölgesi bulunuyor", "Dudaklar okunuyor (dudak okuma modeli)",
         "Whisper sesi yazıya döküyor", "Sonuçlar karşılaştırılıyor"]
SPIN = "◐◓◑◒"


class Window(QtWidgets.QWidget):
    run_job = QtCore.pyqtSignal(str, str, str)
    run_rec = QtCore.pyqtSignal(str, object, object)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.models_ready = self.busy = False
        self.state = "idle"
        self.setWindowTitle("Dudak Okuma")
        self.setStyleSheet(STYLE)
        self.resize(1440, 860)

        # Header
        title = QtWidgets.QLabel("Dudak Okuma")
        title.setStyleSheet("font-size:22px; font-weight:700;")
        sub = QtWidgets.QLabel("TR-LRS v3 · vsr_ft30 · Meclis test WER %34 — ses kullanmadan, yalnız dudaklardan",
                               objectName="muted")
        self.pill = QtWidgets.QLabel()
        header = QtWidgets.QHBoxLayout()
        tcol = QtWidgets.QVBoxLayout()
        tcol.setSpacing(0)
        tcol.addWidget(title)
        tcol.addWidget(sub)
        header.addLayout(tcol)
        header.addStretch(1)
        header.addWidget(self.pill)

        # Left: camera + controls
        self.cam = CameraView()
        self.start_btn = QtWidgets.QPushButton("●  Başlat", objectName="start")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.toggle)
        self.limit_group = QtWidgets.QButtonGroup(self)
        seg = QtWidgets.QHBoxLayout()
        seg.setSpacing(4)
        seg.addWidget(QtWidgets.QLabel("En fazla", objectName="muted"))
        for sec in (10, 20, 30):
            b = QtWidgets.QPushButton(f"{sec} sn", objectName="seg", checkable=True)
            b.setChecked(sec == int(args.seconds) or (sec == 10 and int(args.seconds) not in (10, 20, 30)))
            self.limit_group.addButton(b, sec)
            seg.addWidget(b)
        self.file_btn = QtWidgets.QPushButton("Video dosyası seç…")
        self.file_btn.clicked.connect(self.pick_file)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.start_btn)
        controls.addSpacing(10)
        controls.addLayout(seg)
        controls.addStretch(1)
        controls.addWidget(self.file_btn)
        hint = QtWidgets.QLabel("Boşluk: başlat / durdur  ·  yüz ve ağız açıkça görünsün, el ağızdan uzak  ·  "
                                "model Meclis konuşmasıyla eğitildi; resmi, net cümleler daha iyi okunur",
                                objectName="muted")
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.cam, 1)
        left.addLayout(controls)
        left.addWidget(hint)

        # Right: results
        lr_card, lr_lay = card()
        lr_lay.addWidget(QtWidgets.QLabel("DUDAK OKUMA  ·  yalnız görüntü", objectName="h"))
        self.lr_text = QtWidgets.QLabel(wordWrap=True)
        self.lr_text.setTextFormat(Qt.TextFormat.RichText)
        self.lr_text.setStyleSheet("font-size:21px; font-weight:600;")
        self.lr_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lr_lay.addWidget(self.lr_text)
        lr_card.setStyleSheet(f"QFrame#card {{ border-left:4px solid {BLUE}; }}")

        wh_card, wh_lay = card()
        wh_lay.addWidget(QtWidgets.QLabel("WHISPER LARGE-V3  ·  yalnız ses (referans)", objectName="h"))
        self.wh_text = QtWidgets.QLabel(wordWrap=True)
        self.wh_text.setTextFormat(Qt.TextFormat.RichText)
        self.wh_text.setStyleSheet("font-size:17px;")
        self.wh_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        wh_lay.addWidget(self.wh_text)
        wh_card.setStyleSheet(f"QFrame#card {{ border-left:4px solid {GREEN}; }}")
        self.legend = QtWidgets.QLabel(
            f"<span style='color:{TEXT}'>beyaz</span>: eşleşen kelime &nbsp;·&nbsp; "
            f"<span style='color:#ff7b7f'>kırmızı</span>: yanlış/fazla okunan &nbsp;·&nbsp; "
            f"<span style='color:#6b7280'>gri</span>: kaçırılan", objectName="muted")

        stats = QtWidgets.QHBoxLayout()
        self.s_wer, self.s_cer, self.s_face, self.s_len = Stat("WER"), Stat("CER"), Stat("yüz bulunan kare"), Stat("süre")
        for s in (self.s_wer, self.s_cer, self.s_face, self.s_len):
            stats.addWidget(s)

        strip_card, strip_lay = card()
        strip_lay.addWidget(QtWidgets.QLabel("MODELİN GÖRDÜĞÜ AĞIZ BÖLGESİ", objectName="h"))
        self.strip = QtWidgets.QLabel()
        self.strip.setMinimumHeight(46)
        strip_lay.addWidget(self.strip)

        hist_card, hist_lay = card()
        hrow = QtWidgets.QHBoxLayout()
        hrow.addWidget(QtWidgets.QLabel("GEÇMİŞ KAYITLAR", objectName="h"))
        hrow.addStretch(1)
        self.open_btn = QtWidgets.QPushButton("Klasör")
        self.open_btn.clicked.connect(self.open_folder)
        self.del_btn = QtWidgets.QPushButton("Sil", objectName="danger")
        self.del_btn.clicked.connect(self.delete_selected)
        self.del_all_btn = QtWidgets.QPushButton("Tümünü sil", objectName="danger")
        self.del_all_btn.clicked.connect(self.delete_all)
        for b in (self.open_btn, self.del_btn, self.del_all_btn):
            b.setStyleSheet("padding:5px 10px;")
            hrow.addWidget(b)
        hist_lay.addLayout(hrow)
        self.history = QtWidgets.QListWidget()
        self.history.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.history.currentItemChanged.connect(self.show_history_item)
        hist_lay.addWidget(self.history)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(lr_card)
        right.addWidget(wh_card)
        right.addWidget(self.legend)
        right.addLayout(stats)
        right.addWidget(strip_card)
        right.addWidget(hist_card, 1)
        panel = QtWidgets.QWidget()
        panel.setLayout(right)
        panel.setFixedWidth(580)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(18)
        body.addLayout(left, 1)
        body.addWidget(panel)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(22, 16, 22, 16)
        root.setSpacing(14)
        root.addLayout(header)
        root.addLayout(body, 1)

        self.clear_results("Modeller yükleniyor, birkaç saniye…")
        self.set_pill("Modeller yükleniyor", AMBER)

        # Models in a worker thread.
        self.worker_thread = QtCore.QThread()
        self.worker = Worker(args.run, args.beam, args.whisper)
        self.worker.moveToThread(self.worker_thread)
        self.worker.status.connect(self.on_status)
        self.worker.step.connect(self.on_step)
        self.run_rec.connect(self.worker.process_recording)
        self.worker.ready.connect(self.on_ready)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.run_job.connect(self.worker.process)
        self.worker_thread.started.connect(self.worker.load)
        self.worker_thread.start()

        self.camera = Camera(args.camera)
        self.camera.frame.connect(self.cam.set_frame)
        self.camera.error.connect(lambda m: self.set_pill(m, RED))
        self.camera.start()

        self.ticker = QtCore.QTimer(self, interval=50)
        self.ticker.timeout.connect(self.on_tick)
        self.ticker.start()

        self.load_history()
        self.update_buttons()

    # ---- small helpers ----------------------------------------------------
    def set_pill(self, text, color):
        self.pill.setText(f"●  {text}")
        self.pill.setStyleSheet(f"color:{color}; background:{CARD}; border:1px solid {LINE}; "
                                "border-radius:14px; padding:6px 14px; font-weight:600;")

    def on_status(self, text):
        self.cam.note = text.rstrip("…")
        self.set_pill(text.rstrip("…"), BLUE if self.models_ready else AMBER)

    def on_step(self, n):
        self.step_idx, self.step_t0 = n, time.time()
        self.cam.note = STEPS[n]
        self.set_pill(f"İşleniyor · {n + 1}/{len(STEPS)}", BLUE)
        self.render_steps()

    def begin_steps(self, first):
        self.job_t0 = time.time()
        self.on_step(first)

    def render_steps(self):
        spin = SPIN[int(time.time() * 6) % len(SPIN)]
        rows = []
        for i, name in enumerate(STEPS):
            if i < self.step_idx:
                rows.append(f"<span style='color:{GREEN}'>✓</span>&nbsp;&nbsp;<span style='color:{MUTED}'>{name}</span>")
            elif i == self.step_idx:
                rows.append(f"<span style='color:{BLUE}'>{spin}</span>&nbsp;&nbsp;<b>{name}</b>"
                            f"&nbsp;&nbsp;<span style='color:{MUTED}'>{time.time() - self.step_t0:.0f} sn</span>")
            else:
                rows.append(f"<span style='color:#4b5263'>○&nbsp;&nbsp;{name}</span>")
        self.lr_text.setText("<div style='font-size:15px; font-weight:400; line-height:150%'>"
                             + "<br>".join(rows) + "</div>")
        self.wh_text.setText(f"<span style='color:{MUTED}'>Toplam {time.time() - self.job_t0:.0f} sn · "
                             "10 sn'lik kayıt genelde 15–25 sn sürer. Pencere donmadı, model çalışıyor.</span>")

    def clear_results(self, msg):
        self.lr_text.setText(f"<span style='color:{MUTED}; font-size:15px; font-weight:400'>{msg}</span>")
        self.wh_text.setText(f"<span style='color:{MUTED}'>—</span>")
        for s in (self.s_wer, self.s_cer, self.s_face, self.s_len):
            s.value.setText("—")
        self.strip.clear()

    def fresh(self, msg):
        """New take: drop the previous result and history selection from the panel."""
        self.history.blockSignals(True)
        self.history.clearSelection()
        self.history.setCurrentItem(None)
        self.history.blockSignals(False)
        self.clear_results(msg)

    def set_state(self, state):
        self.state = state
        self.cam.state = state
        self.update_buttons()

    def update_buttons(self):
        recording = self.state in ("countdown", "recording")
        self.start_btn.setProperty("mode", "stop" if recording else "start")
        self.start_btn.setText("■  Durdur" if recording else "●  Başlat")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        idle = self.models_ready and self.state == "idle"
        self.start_btn.setEnabled(idle or recording)
        self.file_btn.setEnabled(idle)
        for b in self.limit_group.buttons():
            b.setEnabled(not recording)
        self.del_btn.setEnabled(self.state == "idle")
        self.del_all_btn.setEnabled(self.state == "idle")

    def on_tick(self):
        if self.state == "recording":
            if time.time() - self.cam.rec_start >= self.cam.rec_limit:
                self.stop_recording()
        if self.state in ("recording", "busy"):
            self.cam.update()
        if self.state == "busy" and hasattr(self, "job_t0"):
            self.render_steps()

    # ---- recording ------------------------------------------------------
    def toggle(self):
        if self.state == "idle":
            self.start_countdown()
        elif self.state == "countdown":
            self.set_state("idle")
            self.set_pill("Hazır", GREEN)
            self.load_history()
        elif self.state == "recording":
            self.stop_recording()

    def start_countdown(self):
        if not self.models_ready or self.state != "idle":
            return
        self.set_state("countdown")
        self.fresh("Hazırlanın… 3-2-1'den sonra konuşun.")
        self.cam.count = 3
        self.set_pill("Hazırlanın", AMBER)
        QtCore.QTimer.singleShot(1000, self.count_tick)

    def count_tick(self):
        if self.state != "countdown":
            return
        self.cam.count -= 1
        if self.cam.count > 0:
            self.cam.update()
            QtCore.QTimer.singleShot(1000, self.count_tick)
            return
        self.out_dir = os.path.join(OUT_ROOT, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.out_dir, exist_ok=True)
        self.wav = os.path.join(self.out_dir, "ses.wav")
        self.audio = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "pulse", "-i", "default",
             "-ac", "1", "-ar", "16000", "-t", "65", self.wav], stdin=subprocess.PIPE)
        self.camera.frames = []
        self.camera.recording = True
        self.cam.rec_start = time.time()
        self.cam.rec_limit = float(self.limit_group.checkedId())
        self.set_state("recording")
        self.set_pill("Kayıt — konuşun", RED)
        self.clear_results("Kayıt sürüyor… bitince Durdur'a basın, sonuç burada görünecek.")

    def stop_recording(self):
        if self.state != "recording":
            return
        self.camera.recording = False
        frames, self.camera.frames = self.camera.frames, []
        try:
            self.audio.communicate(b"q", timeout=5)
        except Exception:  # noqa: BLE001
            self.audio.kill()
        if len(frames) < 25:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self.on_failed("Kayıt çok kısa (en az 1 sn konuşun).")
            return
        self.set_state("busy")
        fps = Fraction((len(frames) - 1) / (frames[-1][0] - frames[0][0])).limit_denominator(1000)
        self.begin_steps(0)
        self.run_rec.emit(self.out_dir, [f for _, f in frames], fps)

    # ---- file -----------------------------------------------------------
    def pick_file(self):
        if not self.models_ready or self.state != "idle":
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Video seç", os.path.expanduser("~"), "Videolar (*.mp4 *.mkv *.webm *.mov *.avi);;Tümü (*)")
        if not path:
            return
        self.out_dir = os.path.join(OUT_ROOT, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_dosya")
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "kaynak.txt"), "w") as f:
            f.write(path + "\n")
        self.set_state("busy")
        self.fresh("")
        self.begin_steps(1)
        self.run_job.emit(path, path, self.out_dir)

    # ---- results --------------------------------------------------------
    def on_ready(self):
        self.models_ready = True
        self.set_pill("Hazır", GREEN)
        if not self.history.count():
            self.clear_results("Başlat'a basın, 3-2-1'den sonra konuşun, bitince Durdur'a basın.")
        self.update_buttons()

    def show_result(self, r, strip_path):
        ref, hyp = r["whisper"].split(), r["lip_reading"].split()
        rt, ht = align(ref, hyp)
        self.lr_text.setText(words_html(hyp, ht, "#ff7b7f") if hyp else f"<span style='color:{MUTED}'>(boş)</span>")
        self.wh_text.setText(words_html(ref, rt, "#6b7280") if ref
                             else f"<span style='color:{MUTED}'>(konuşma duyulmadı)</span>")
        self.s_wer.value.setText(f"%{r['wer'] * 100:.0f}" if r.get("wer") is not None else "—")
        self.s_cer.value.setText(f"%{r['cer'] * 100:.0f}" if r.get("cer") is not None else "—")
        self.s_face.value.setText(f"%{r['face_found'] * 100:.0f}")
        self.s_len.value.setText(f"{r['seconds']:.1f} sn")
        pix = QtGui.QPixmap(strip_path)
        self.strip.setPixmap(pix.scaledToWidth(540, Qt.TransformationMode.SmoothTransformation)
                             if not pix.isNull() else QtGui.QPixmap())

    def on_done(self, r):
        self.on_step(4)
        self.set_state("idle")
        self.set_pill("Hazır — yeni deneme için Başlat", GREEN)
        if r["face_found"] < 0.8:
            self.set_pill("Yüz bazı karelerde bulunamadı", AMBER)
        self.load_history(select=r["out_dir"])

    def on_failed(self, msg):
        self.set_state("idle")
        self.set_pill(msg, RED)
        self.clear_results(msg)

    # ---- history --------------------------------------------------------
    def load_history(self, select=None):
        self.history.blockSignals(True)
        self.history.clear()
        if os.path.isdir(OUT_ROOT):
            for name in sorted(os.listdir(OUT_ROOT), reverse=True):
                d = os.path.join(OUT_ROOT, name)
                try:
                    r = json.load(open(os.path.join(d, "sonuc.json"), encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                when = name[:16].replace("_", "  ").replace("-", ".", 2).replace("-", ":")
                kind = "dosya" if name.endswith("_dosya") else "kamera"
                wer = f"WER %{r['wer'] * 100:.0f}" if r.get("wer") is not None else "WER —"
                text = r["lip_reading"][:38] + ("…" if len(r["lip_reading"]) > 38 else "")
                item = QtWidgets.QListWidgetItem(f"{when}   ·   {kind}   ·   {wer}\n{text}")
                item.setData(Qt.ItemDataRole.UserRole, d)
                self.history.addItem(item)
                if select and os.path.samefile(d, select):
                    self.history.setCurrentItem(item)
        self.history.blockSignals(False)
        cur = self.history.currentItem()
        if cur is None and self.history.count():
            self.history.setCurrentRow(0)
            cur = self.history.currentItem()
        if cur is not None:
            self.show_history_item(cur)
        elif self.models_ready:
            self.clear_results("Başlat'a basın, 3-2-1'den sonra konuşun, bitince Durdur'a basın.")

    def show_history_item(self, item, _prev=None):
        if item is None:
            return
        d = item.data(Qt.ItemDataRole.UserRole)
        try:
            r = json.load(open(os.path.join(d, "sonuc.json"), encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.show_result(r, os.path.join(d, "strip.png"))

    def _remove(self, dirs):
        root = os.path.realpath(OUT_ROOT)
        for d in dirs:
            if os.path.dirname(os.path.realpath(d)) == root:  # never outside the recordings folder
                shutil.rmtree(d, ignore_errors=True)
        self.load_history()

    def delete_selected(self):
        items = self.history.selectedItems()
        if not items:
            return
        if QtWidgets.QMessageBox.question(self, "Sil", f"{len(items)} kayıt silinsin mi?") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            self._remove([i.data(Qt.ItemDataRole.UserRole) for i in items])

    def delete_all(self):
        n = self.history.count()
        if n and QtWidgets.QMessageBox.question(self, "Tümünü sil", f"Tüm {n} kayıt silinsin mi?") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            self._remove([self.history.item(i).data(Qt.ItemDataRole.UserRole) for i in range(n)])

    def open_folder(self):
        item = self.history.currentItem()
        path = item.data(Qt.ItemDataRole.UserRole) if item else OUT_ROOT
        os.makedirs(path, exist_ok=True)
        subprocess.Popen(["xdg-open", path])

    # ---- keys / close ---------------------------------------------------
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self.toggle()
        elif e.key() == Qt.Key.Key_Delete:
            self.delete_selected()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        if self.state == "recording":
            self.camera.recording = False
            self.audio.kill()
        self.camera.running = False
        self.camera.wait(2000)
        self.worker_thread.quit()
        self.worker_thread.wait(2000)
        e.accept()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN, help="eğitilmiş model klasörü (tr-vsr-model)")
    ap.add_argument("--seconds", type=float, default=10.0, help="en uzun kayıt (10, 20 ya da 30)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--whisper", default="large-v3")
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Dudak Okuma")
    w = Window(args)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
