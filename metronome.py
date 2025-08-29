#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metronome GUI
BPM・拍子・秒数を指定してメトロノームをプレビュー再生＆WAV出力するGUIツール。

依存:
- numpy (必須)
- simpleaudio (任意：プレビュー再生に使用。未導入でもWAV出力は可能)

author: @sadmb
"""

import io
import os
import wave
import threading
import platform
import functools
import tempfile
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ---- プレビュー用バックエンド検出 ----
IS_WINDOWS = (platform.system() == "Windows")
if IS_WINDOWS:
    import winsound

try:
    import simpleaudio as sa  # 任意（Windowsでは使わない）
    _HAS_SIMPLEAUDIO = True
except Exception:
    sa = None
    _HAS_SIMPLEAUDIO = False


# ===== 音源 =====
def make_click(sr: int, length_s: float, freq: float, amp: float) -> np.ndarray:
    length_s = max(1e-4, float(length_s))
    t = np.linspace(0.0, length_s, int(sr * length_s), endpoint=False, dtype=np.float32)
    click = np.sin(2.0 * np.pi * freq * t, dtype=np.float32)
    decay_rate = 10.0 + 0.002 * freq
    env = np.exp(-decay_rate * t, dtype=np.float32)
    attack = np.minimum(1.0, (t / max(1e-3, length_s * 0.08))).astype(np.float32)
    return (click * env * attack * amp).astype(np.float32)


def build_metronome(
    sr: int, bpm: float, time_signature: Tuple[int, int], duration_s: float,
    accent_hz: float = 2000.0, mid_hz: float = 1600.0, beat_hz: float = 1200.0,
    strong_len: float = 0.030, weak_len: float = 0.020, master_gain: float = 0.9,
) -> np.ndarray:
    n_num, n_den = time_signature
    total_samples = int(sr * duration_s)
    buf = np.zeros(total_samples, dtype=np.float32)

    # 分母に合わせて拍間隔をスケール（例: 7/8 → ♪間隔）
    note_len_factor = 4.0 / float(n_den)
    sec_per_beat = (60.0 / float(bpm)) * note_len_factor

    strong_len = min(strong_len, sec_per_beat * 0.45)
    weak_len   = min(weak_len,   sec_per_beat * 0.45)

    # 6/8, 9/8, 12/8 は 1 と 4 に弱アクセント
    use_compound_mid = (n_den == 8 and n_num % 3 == 0)

    beat_index = 0
    t = 0.0
    while t < duration_s:
        start = int(t * sr)
        if start >= total_samples:
            break

        beat_in_bar = beat_index % n_num
        if beat_in_bar == 0:
            clk = make_click(sr, strong_len, accent_hz, amp=1.0)
        elif use_compound_mid and (beat_in_bar % 3 == 0):
            clk = make_click(sr, weak_len, mid_hz, amp=0.8)
        else:
            clk = make_click(sr, weak_len, beat_hz, amp=0.7)

        end = min(total_samples, start + clk.size)
        if end > start:
            buf[start:end] += clk[: end - start]

        beat_index += 1
        t = beat_index * sec_per_beat

    mx = np.max(np.abs(buf)) if np.any(buf) else 1.0
    if mx > 1.0:
        buf /= mx
    buf *= master_gain
    np.clip(buf, -1.0, 1.0, out=buf)
    return buf


def float32_to_int16(x: np.ndarray) -> np.ndarray:
    return np.int16(np.clip(x, -1.0, 1.0) * 32767.0)


def write_wav(path: str, sr: int, audio: np.ndarray) -> None:
    pcm = float32_to_int16(audio)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ===== GUI =====
@dataclass
class MetronomeParams:
    bpm: float
    ts_num: int
    ts_den: int
    duration_s: float
    sr: int


class MetronomeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Metronome WAV Generator")

        # 再生状態
        self._play_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        # Windows: 一時WAVファイルのパス
        self._tmp_wav_path: Optional[str] = None
        # 他OS: simpleaudio の PlayObject
        self._play_obj = None

        self._build_widgets()
        self.after(0, self._set_minsize_to_fit)

    def _set_minsize_to_fit(self):
        self.update_idletasks()
        self.minsize(self.winfo_reqwidth() + 24, self.winfo_reqheight() + 24)

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="BPM").grid(row=0, column=0, sticky="w")
        self.var_bpm = tk.DoubleVar(value=120.0)
        ttk.Spinbox(frm, from_=10, to=400, increment=1, textvariable=self.var_bpm, width=8)\
            .grid(row=0, column=1, sticky="w")

        ttk.Label(frm, text="Time Signature").grid(row=1, column=0, sticky="w")
        self.var_ts_num = tk.IntVar(value=4)
        self.var_ts_den = tk.IntVar(value=4)
        ts_box = ttk.Frame(frm)
        ttk.Spinbox(ts_box, from_=1, to=32, increment=1, textvariable=self.var_ts_num, width=4)\
            .pack(side="left")
        ttk.Label(ts_box, text="/").pack(side="left", padx=(6, 6))
        ttk.Spinbox(ts_box, from_=1, to=32, increment=1, textvariable=self.var_ts_den, width=4)\
            .pack(side="left")
        ts_box.grid(row=1, column=1, columnspan=3, sticky="w")

        ttk.Label(frm, text="Duration (sec)").grid(row=2, column=0, sticky="w")
        self.var_dur = tk.DoubleVar(value=10.0)
        ttk.Spinbox(frm, from_=1, to=600, increment=1, textvariable=self.var_dur, width=8)\
            .grid(row=2, column=1, sticky="w")

        ttk.Label(frm, text="Sample Rate").grid(row=3, column=0, sticky="w")
        self.var_sr = tk.IntVar(value=48000)
        ttk.Combobox(frm, textvariable=self.var_sr, width=10,
                     values=[44100, 48000, 88200, 96000], state="readonly")\
            .grid(row=3, column=1, sticky="w")

        frm_btn = ttk.Frame(frm)
        frm_btn.grid(row=4, column=0, columnspan=4, sticky="w", pady=(12, 0))
        self.btn_preview = ttk.Button(frm_btn, text="▶︎ Preview", command=self.on_preview)
        self.btn_stop    = ttk.Button(frm_btn, text="■ Stop", command=self.on_stop, state="disabled")
        self.btn_export  = ttk.Button(frm_btn, text="💾 Export WAV", command=self.on_export)
        self.btn_preview.grid(row=0, column=0, padx=4)
        self.btn_stop.grid(row=0, column=1, padx=4)
        self.btn_export.grid(row=0, column=2, padx=8)

        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(frm, textvariable=self.var_status, foreground="#444")\
            .grid(row=5, column=0, columnspan=4, sticky="w", pady=(12, 0))

        if not _HAS_SIMPLEAUDIO and not IS_WINDOWS:
            self.btn_preview.configure(state="disabled")
            self.var_status.set("プレビュー不可（pip install simpleaudio で有効化）")

    # ---- 共通 ----
    def _get_params(self) -> Optional[MetronomeParams]:
        try:
            bpm = float(self.var_bpm.get())
            ts_num = int(self.var_ts_num.get())
            ts_den = int(self.var_ts_den.get())
            dur = float(self.var_dur.get())
            sr = int(self.var_sr.get())
        except Exception:
            messagebox.showerror("Error", "数値の入力が不正です。")
            return None
        if bpm <= 0 or dur <= 0:
            messagebox.showerror("Error", "BPM と Duration は正の値を入力してください。")
            return None
        if ts_num <= 0 or ts_den not in (1, 2, 4, 8, 16, 32):
            messagebox.showerror("Error", "拍子は 分子>0、分母は 1/2/4/8/16/32 のいずれかです。")
            return None
        if sr not in (44100, 48000, 88200, 96000):
            messagebox.showerror("Error", "Sample Rate が不正です。")
            return None
        return MetronomeParams(bpm=bpm, ts_num=ts_num, ts_den=ts_den, duration_s=dur, sr=sr)

    # ---- プレビュー ----
    def on_preview(self):
        if self._play_thread and self._play_thread.is_alive():
            return
        p = self._get_params()
        if not p:
            return

        self._set_ui_playing(True, "Rendering...")

        def _worker():
            try:
                audio = build_metronome(
                    sr=p.sr, bpm=p.bpm,
                    time_signature=(p.ts_num, p.ts_den),
                    duration_s=p.duration_s,
                )
                pcm = float32_to_int16(audio).tobytes()

                if IS_WINDOWS:
                    # 一時WAVファイルへ書き出して winsound で非同期再生
                    fd, path = tempfile.mkstemp(prefix="metronome_preview_", suffix=".wav")
                    os.close(fd)
                    with wave.open(path, "wb") as wf:
                        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(p.sr)
                        wf.writeframes(pcm)
                    self._tmp_wav_path = path
                    winsound.PlaySound(path,
                        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
                    est_ms = int((len(pcm) / 2) / p.sr * 1000) + 100
                    self.after(est_ms, self._finish_if_still_playing)
                else:
                    self._play_obj = sa.play_buffer(pcm, 1, 2, p.sr)  # type: ignore
                    self.after(50, self._poll_simpleaudio)

                # ステータス更新（lambda を使わない）
                self.after(0, self.var_status.set, "Playing preview...")

            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.after(0, functools.partial(self._finish_preview, error=msg))

        self._stop_flag.clear()
        self._play_thread = threading.Thread(target=_worker, daemon=True)
        self._play_thread.start()

    def _finish_if_still_playing(self):
        if not self._stop_flag.is_set():
            self._finish_preview()

    def _poll_simpleaudio(self):
        if self._stop_flag.is_set():
            try:
                if self._play_obj and self._play_obj.is_playing():
                    self._play_obj.stop()
            except Exception:
                pass
            self._finish_preview()
            return
        if self._play_obj is None:
            self.after(50, self._poll_simpleaudio)
            return
        try:
            if not self._play_obj.is_playing():
                self._finish_preview()
                return
        except Exception:
            self._finish_preview()
            return
        self.after(50, self._poll_simpleaudio)

    def on_stop(self):
        self._stop_flag.set()
        if IS_WINDOWS:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            finally:
                if self._tmp_wav_path and os.path.exists(self._tmp_wav_path):
                    try:
                        os.remove(self._tmp_wav_path)
                    except Exception:
                        pass
                    self._tmp_wav_path = None
                self._finish_preview()
        # 他OSは _poll_simpleaudio 側で停止処理

    def _finish_preview(self, error: Optional[str] = None):
        self._stop_flag.clear()
        # Windows: 一時ファイルを削除
        if IS_WINDOWS and self._tmp_wav_path:
            try:
                if os.path.exists(self._tmp_wav_path):
                    os.remove(self._tmp_wav_path)
            except Exception:
                pass
            self._tmp_wav_path = None
        # 他OS: オブジェクト解放
        self._play_obj = None

        if error:
            self._set_ui_playing(False, f"Error: {error}")
            messagebox.showerror("Error", error)
        else:
            self._set_ui_playing(False, "Ready")

    # ---- 書き出し ----
    def on_export(self):
        p = self._get_params()
        if not p:
            return
        default_name = f"metronome_{int(p.bpm)}_{p.ts_num}-{p.ts_den}_{int(p.duration_s)}s.wav"
        path = filedialog.asksaveasfilename(
            title="WAVを書き出し",
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[("WAV", "*.wav")],
        )
        if not path:
            return
        try:
            self._set_ui_busy(True, "Rendering...")
            audio = build_metronome(sr=p.sr, bpm=p.bpm,
                                    time_signature=(p.ts_num, p.ts_den),
                                    duration_s=p.duration_s)
            write_wav(path, p.sr, audio)
            self._set_ui_busy(False, f'WAVを書き出しました: {path}')
        except Exception as e:
            self._set_ui_busy(False, f"Error: {e}")
            messagebox.showerror("Error", str(e))

    # ---- UI ----
    def _set_ui_busy(self, busy: bool, status: str):
        self.var_status.set(status)
        state = "disabled" if busy else "normal"
        self.btn_export.configure(state=state)
        self.btn_preview.configure(state=state)

    def _set_ui_playing(self, playing: bool, status: str):
        self.var_status.set(status)
        self.btn_preview.configure(state="disabled" if playing else "normal")
        self.btn_stop.configure(state="normal" if playing else "disabled")
        self.btn_export.configure(state="disabled" if playing else "normal")


if __name__ == "__main__":
    app = MetronomeGUI()
    app.mainloop()
