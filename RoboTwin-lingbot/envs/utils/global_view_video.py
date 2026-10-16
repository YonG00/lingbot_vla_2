"""评测专用全景录像：按物理时间采帧，不接触策略观测、物理步进或 RNG。"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


def look_at_matrix(position, target):
    """SAPIEN 相机局部轴为 forward / left / up；构造正交单位旋转矩阵。"""
    def unit(v):
        norm = math.sqrt(sum(x * x for x in v))
        if not math.isfinite(norm) or norm < 1e-9:
            raise ValueError("全景相机 position/target 必须有限、不同且不能竖直重合")
        return [x / norm for x in v]

    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    if len(position) != 3 or len(target) != 3 or not all(math.isfinite(x) for x in (*position, *target)):
        raise ValueError("全景相机 position/target 必须是三个有限坐标")
    forward = unit([t - p for p, t in zip(position, target)])
    left = unit(cross([0, 0, 1], forward))
    up = cross(forward, left)
    return [[forward[i], left[i], up[i], position[i]] for i in range(3)] + [[0, 0, 0, 1]]


class GlobalViewRecorder:
    """每个回合一个 encoder。after_step 只能在已完成的物理步后调用。

    视频时间按 simulation time 计算，模型推理/加载等待不写入视频。
    fps 不得超过物理频率；不复制旧帧冒充高帧率。
    """

    def __init__(self, path, capture_rgb, *, timestep, fps=60, width=1280, height=720,
                 preview_writer=None, camera_config=None):
        if (not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 120
                or not math.isfinite(timestep) or timestep <= 0 or fps * timestep > 1):
            raise ValueError("全景 FPS 必须为 1..120 整数，且不能高于物理步频")
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 or v % 2 for v in (width, height)):
            raise ValueError("全景录像宽高必须为正偶数（yuv420p）")
        self.path = Path(path)
        self.capture_rgb = capture_rgb
        self.preview_writer = preview_writer
        self.camera_config = camera_config or {}
        self.timestep, self.fps = timestep, fps
        self.width, self.height = width, height
        self.frames = self.physics_steps = 0
        self.last_capture_step = 0
        self.process = None
        self.error_stream = None
        self.closed = False

    def start(self):
        if self.process is not None or self.closed:
            raise RuntimeError("全景录像不能重复启动")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.error_stream = self.path.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen(
            ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
             "-video_size", f"{self.width}x{self.height}", "-framerate", str(self.fps), "-i", "-",
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.error_stream)
        self._write_frame()  # t=0，并保存第一帧供确认构图。
        print(f"[global_view] 开始：{self.width}x{self.height} {self.fps} FPS（仿真时间） {self.path}", flush=True)

    def _write_frame(self):
        frame = self.capture_rgb()
        if tuple(frame.shape) != (self.height, self.width, 3) or str(frame.dtype) != "uint8":
            raise ValueError("全景相机必须返回与视频尺寸一致的 uint8 RGB 图像")
        if self.frames == 0 and self.preview_writer is not None:
            self.preview_writer(frame, self.path.with_suffix(".preview.png"))
        self.process.stdin.write(frame.tobytes())
        self.frames += 1
        self.last_capture_step = self.physics_steps

    def after_step(self):
        if self.process is None or self.closed:
            raise RuntimeError("全景录像尚未启动或已关闭")
        self.physics_steps += 1
        # 用整数帧号定位下一帧，避免每帧 round(250/60) 导致累计漂移。
        if self.physics_steps * self.timestep * self.fps + 1e-9 >= self.frames:
            self._write_frame()

    def close(self, result="interrupted"):
        if self.closed:
            return
        error = None
        # Success can occur between scheduled frames. Preserve the terminal pose
        # with at most one extra frame, rather than silently losing the outcome.
        if result in ("success", "failure") and self.process is not None and self.last_capture_step != self.physics_steps:
            try:
                self._write_frame()
            except Exception as exc:
                error = f"全景末帧写入失败：{exc}"
        self.closed = True
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    error = "ffmpeg 管道提前关闭"
                try:
                    code = self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                    code = -1
                if code != 0:
                    error = f"ffmpeg 退出码 {code}；查看 {self.path.with_suffix('.ffmpeg.log')}"
        finally:
            if self.error_stream is not None:
                self.error_stream.close()
        metadata = dict(fps=self.fps, width=self.width, height=self.height, frames=self.frames,
                        physics_steps=self.physics_steps, timestep=self.timestep,
                        simulation_seconds=self.physics_steps*self.timestep,
                        video_seconds=self.frames/self.fps, camera=self.camera_config,
                        result=result, encoder_ok=error is None and self.process is not None, error=error)
        self.path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
        if error:
            raise RuntimeError(error)
        if self.process is not None:
            print(f"[global_view] 完成：{self.frames} 帧，仿真 {metadata['simulation_seconds']:.3f}s，"
                  f"视频 {metadata['video_seconds']:.3f}s，result={result}", flush=True)
