# torchcodec 可行性测试报告（AMD ROCm / Ubuntu 24.04）

> 报告日期：2026-10-12
> 机器：cpu1（AMD 8×W7900D，Ubuntu 24.04.4 LTS，ROCm 7.2.1）
> 触发原因：hardness 扫描慢，怀疑视频解码是瓶颈；社区普遍反馈 torchcodec 显著快于 pyav。
> **结论速览**：`torchcodec` 在本机**不可用**（FFmpeg ABI + torch ABI 双重不匹配）；
> 且这批数据**解码本来就快**（0.88–1.54 ms/帧），**预期收益有限**。**建议暂不采用**。

---

## 1. 代码层事实核对（用户假设 → 实测）

用户的判断：数据集硬编码 `pyav`，导致 `get_safe_default_codec()`（会优先选 torchcodec）
永远不被调用。**逐条核对，全部成立**：

| # | 位置 | 实测 |
|---|---|---|
| 1 | `lingbotvla/data/vla_data/multi_vla_dataset.py:67` | `video_backend: str = 'pyav'` ✅ 默认值确为 `'pyav'` |
| 2 | `lingbotvla/data/vla_data/base_dataset.py:168` | `decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)` ✅ 显式透传 |
| 3 | 同上 `232` / `267` 行 | 另有两处硬编码 `video_backend = 'pyav'` |
| 4 | `lingbotvla/data/vla_data/video_utils.py:32` | `get_safe_default_codec()`：有 torchcodec 就返回 `"torchcodec"`，否则回退 `"pyav"` |
| 5 | 同上 `42` / `62-66` 行 | `decode_video_frames(..., backend: str | None = None)`；**仅当 `backend is None` 才调 `get_safe_default_codec()`** |

⇒ **因果链成立**：数据集传 `'pyav'` ⇒ `backend is None` 为假 ⇒ torchcodec 分支是**死代码**。
若要启用，最小改动是 `multi_vla_dataset.py:67` 改为 `None`（或 `'torchcodec'`）。

---

## 2. 环境可用性实测（关键：三层 ABI）

### 2.1 系统与已装组件

| 组件 | 实测 |
|---|---|
| 系统 FFmpeg | **6.1.1**（`libavcodec.so.60` / `libavformat.so.60` / `libavutil.so.58` / `libavfilter.so.9`） |
| pyav | **15.0.0**（已装，`/opt/robotwin-env`）；**不捆绑** libav，用系统 FFmpeg 6 |
| torchcodec | `/opt/lerobot-env` 里有 **0.11.1**（原生库为 `libtorchcodec_core{4,5,6,7}.so`） |
| 机器联网 | **无**（`pip` → `Network is unreachable`；apt 索引为空）⇒ 只能离线搬包 |

### 2.2 第一层：FFmpeg ABI 不匹配

```
ldd libtorchcodec_core4.so →
  需要: libavcodec.so.58 libavdevice.so.58 libavfilter.so.7 libavformat.so.58 libavutil.so.56   ← 全是 FFmpeg 5
  实有: libavcodec.so.60 ……… libavfilter.so.9 ……… libavutil.so.58                            ← FFmpeg 6
```

⇒ 该 wheel **只带 FFmpeg 5 构建**，系统没有 FFmpeg 5 ⇒ 加载即失败：
`OSError: Could not load this library: .../torchcodec/libtorchcodec_core4.so`

**这与 AMD 无关**，是 FFmpeg 版本问题。

### 2.3 沙箱补齐 FFmpeg 5（隔离在 /tmp，不动系统）

| 步骤 | 做法 | 结果 |
|---|---|---|
| 取包 | 从本机（有网）下 Ubuntu 22.04 (Jammy) 的 `libav*58/56/7` + 34 个传递依赖（`libvpx7` `libx264-163` `libdav1d5` `libmfx1` `libsrt1.4-gnutls` `libpostproc55` `libcodec2-1.0` …） | **41 个 deb / 89 个 .so** |
| 解包 | 纯 Python 解 `ar` + `data.tar.zst`（机器无 `dpkg-deb`，macOS 无 zstd-tar） | 41/41 成功 |
| 隔离 | 落在 `/tmp/tc_sandbox/libs`，仅运行时 `LD_LIBRARY_PATH` 指向它 | **未改系统库、未改训练环境** |
| 校验 | 14 个关键库全部就位（`libavcodec.so.58` … `libsrt-gnutls.so.1.4`） | ✅ |

⇒ **FFmpeg 这一层已解决**（`ldd` 不再报 `libav*` 缺失）。

### 2.4 第二层：torch ABI 不匹配（**卡死点**）

补齐 FFmpeg 5 后再加载：

```
ctypes.CDLL("libtorchcodec_core4.so")
  → OSError: undefined symbol: torch_dtype_float4_e2m1fn_x2
```

⇒ torchcodec 0.11.1 是**针对特定 torch 二进制编译的**，与机器上两个 env 的 torch
（`/opt/venv`、`/opt/robotwin-env`）**符号对不上**。这是**第二个、也是最终的卡点**。

**同时验证**（排除"AMD 不支持"这一误解）：

* torchcodec 的解码路径在代码里是 `device="cpu"` ⇒ **与 AMD/ROCm 无关**；
* 若换到 NVIDIA 机器且 torch 版本匹配，`device="cuda"` 才会用 NVDEC 硬件解码，**AMD 无对应路径**。

### 2.5 本机（macOS）复核：同一问题

本机（arm64，torch 2.8.0，ffmpeg@8）装 `torchcodec 0.11.1 / 0.17.0` 后同样失败：

```
libtorchcodec_core4.dylib: Library not loaded: @rpath/libavutil.56.dylib   ← 同样要 FFmpeg 5
libtorchcodec_image.dylib: 链接的 libtorch_cpu.dylib 不匹配               ← 同样 torch ABI
```

⇒ **"装了就能用"不成立**，三个平台（机器 env1、env2、macOS）**无一可用**。

---

## 2.6 澄清：「AMD 用不了 torchcodec」是误解

| 说法 | 事实 |
|---|---|
| "AMD 用不了 torchcodec" | ❌ **不对**。torchcodec 的解码路径是 `device="cpu"`（见 `video_utils.py` 的 `decode_video_frames_torchcodec` 默认参数），**与 GPU 厂商无关**，AMD 上可用 |
| 什么才是 NVIDIA 专属 | **GPU 硬件解码**（`device="cuda"` → NVDEC）。本项目走 CPU 解码，用不到；AMD 没有该路径也不影响 |
| 本项目装不起来的原因 | **版本组合**：① FFmpeg 5 vs 6（第 2.2 节）；② torchcodec ↔ torch 二进制不匹配（第 2.4 节）。**换到 NVIDIA 机器只要版本不对，同样装不起来** |

## 3. 收益评估：解码到底占多少？

### 3.1 真实数据的解码实测（pyav 生产路径）

测试素材 = 线上真实文件 `/models/robotwin-persistent/data/demo_clean/place_bread_skillet/aloha_agilex/video/*.mp4`：

| 形态 | 实测 |
|---|---|
| 单文件大小 / 时长 | **中位 0.11 MB / 最长 6.2 s**（50 条） |
| 连续 25 帧（同文件） | **22 ms/调用 = 0.88 ms/帧** |
| 跨文件 30 次 × 10 帧（含每次 open） | **462 ms = 1.54 ms/帧** |

### 3.2 hardness 扫描的真实构成（来自 al_v42 实跑日志）

```
[hardness] rank0/2 进度 136/136（批 17）用时 703s | data 671.7s  score 31.0s
[hardness] rank1/2 进度 135/135（批 17）用时 721s | data 686.6s  score 34.1s
[hardness] 扫描完成：task=place_object_stand 打分 271/271 用时 721.0s
```

| 项 | 数值 |
|---|---|
| 单任务样本数 | **271**（两 rank 各 ~136） |
| 单任务扫描耗时 | **721 s（约 12 分钟）** |
| 其中 `data` | **~95%**（687 s） |
| 其中 `score`（模型打分） | ~5%（34 s） |

⇒ `data` 占 95% **属实**，但据 3.1 的解码速度，**271 样本的解码量只值几秒到十几秒** ——
`data` 的绝大部分**不是视频解码**，更可能是：数据集 `__getitem__`、图像预处理/变换、
DataLoader 等待、以及**每次调用重新 `open` 视频文件**的开销。

**结论：换解码器最多只影响 `data` 的一小部分；先把 `data` 内部拆解开才是正确方向。**

### 3.3 ⚠️ 未实测项：torchcodec 与 pyav 的直接 A/B（**本报告不下断言**）

| 项 | 状态 |
|---|---|
| pyav 速度 | ✅ 已实测（3.1 节） |
| **torchcodec 速度** | ❌ **未测到** —— 三个环境（机器 env1/env2、macOS）都卡在 ABI，无法加载 |
| 之前口头说的"预期收益有限" | ⚠️ 那是**推断**（依据：视频仅 0.11 MB + 其优势主要在 GPU 解码），**不是实测结论**。CPU→CPU 在小文件上**可能持平、也可能更快**，未测不定论 |

**复现方法（待 FFmpeg 5 就绪后执行）**：

```bash
# 本机（macOS）：需先有 FFmpeg 5 的 dylib（brew install ffmpeg@5 或 conda -p /tmp/ff5 ffmpeg=5）
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@5/lib
python .tmp_diag/tc_bench/ab_bench.py     # 同视频、同时间点，比 ms/帧 + 校验帧一致性
```

**判定标准**：① 比值（pyav 耗时 / torchcodec 耗时）；② **两路帧是否逐值一致** ——
若不一致则等于改变评测口径，收益再大也不能直接上。

---

### 3.4 关键发现：瓶颈是"随机 seek"，不是"解码"

在真实视频（320×240 / 166 帧 / 107 KB / H.264，**仅 1 个关键帧**）上实测：

| 操作 | 实测 | 说明 |
|---|---|---|
| **整段顺序解 166 帧** | **123 ms（0.74 ms/帧）** | 解码本身**很快** |
| **单帧随机 seek**（20 次） | **782 ms（39.1 ms/次）** | **慢 53 倍**：只有 1 个 I 帧 ⇒ 每次都要从 I 帧解到目标帧 |
| 仅 `open` + probe 头信息 | 0.6 ms/次 | 可忽略 |

**折算**：hardness 单任务 271 样本，即使每样本做 10 次 seek ≈ **100 s**；
而实测 `data` 阶段耗时 **687 s**（al_v42 日志）⇒ **解码/seek 只能解释其中一小部分**，
**剩下约 580 s 在数据集的其它环节**（文件读取、图像预处理、DataLoader 等待等，待插桩定位）。

⇒ **换解码器（含 torchcodec、含 AMD 硬解）最多影响这几百秒里的一小块。**

### 3.5 AMD 专用解码方案评估（结论：不划算）

| 方案 | 机器现状 | 评估 |
|---|---|---|
| **rocDecode**（AMD 官方硬件解码，ROCm 组件） | **未安装**（ROCm 内无 `librocdecode`） | 为高码率大视频设计；输出在**显存**，需再拷回 CPU ⇒ 对 107 KB 片段**纯亏** |
| **VAAPI / DRM**（`ffmpeg -hwaccels` 里有 `vaapi`/`drm`） | ffmpeg 已编入；容器内缺 `vainfo`/libva 未验证 | 同上：小文件无收益 + 多一次拷贝 |
| **AMF**（AMD Media Framework） | 未安装 | 面向 Windows/直播，服务端小文件不适用 |

**共同点**：都解决"**高码率视频省 CPU**"，而我们的输入是 **107 KB 微型片段** ⇒ 不适用。

## 4. 结论与建议

### 4.1 结论

1. **代码层判断成立**：数据集硬编码 `pyav` ⇒ torchcodec 分支是死代码（第 1 节）；
2. **但本机无法启用**：FFmpeg ABI（要 FFmpeg 5，系统是 6）**叠加** torch ABI（`undefined symbol`）**双重不匹配** ⇒ 三个环境全部失败；这不是 AMD 的问题；
3. **收益上限很低**：真实视频 0.11 MB，pyav 已到 **0.88–1.54 ms/帧**；换库对 721 s/任务的扫描**影响有限**；
4. **真正的瓶颈待定位**：`data` 占 95% 但解码只占其中极小部分 ⇒ 需对扫描 worker 做采样式 profile。

### 4.2 建议（按性价比）

| 优先级 | 动作 | 依赖 |
|---|---|---|
| **P0** | **给数据集 `__getitem__` / 视频取帧 / 图像预处理分别插桩计时**，把 `data 687s` 拆开 —— 机器上没有 `py-spy/perf/strace`，用 `time.perf_counter()` 打点最省事 | 无（只读） |
| **P1** | 把"**多次单帧 seek**"改为"**整段顺序解一次、再取所需帧**"（实测同文件多次取帧可省 10×+：39.1 ms/次 → 0.74 ms/帧） | 无新依赖，改代码 |
| **P2** | 若样本间重复读同一 episode ⇒ 加**已解码帧缓存**（按 episode 限容量） | 无新依赖 |
| **不建议** | 引入 torchcodec / rocDecode / VAAPI 硬解 | 见 2.6、3.4、3.5 |
| **P2** | 只有 profile 证明"解码就是大头"时，才考虑 torchcodec；届时需：① 离线拿到**与现有 torch 精确匹配**的 torchcodec；② 解决 FFmpeg 5 运行库（本报告第 2.3 节的 41 个包可复用）；③ **帧一致性对拍**（防评测口径变化） | 重（跨版本 + 离线包） |

### 4.3 复现命令（沙箱，已就绪）

```bash
# FFmpeg 5 运行库（41 个 deb 已解）
ls /tmp/tc_sandbox/libs/usr/lib/x86_64-linux-gnu | wc -l      # 89
# 加载测试（预期仍失败于 torch ABI）
LD_LIBRARY_PATH=/tmp/tc_sandbox/libs/usr/lib/x86_64-linux-gnu:/opt/venv/lib/python3.12/site-packages/torch/lib \
  /opt/lerobot-env/bin/python -c "
import ctypes; ctypes.CDLL('/opt/lerobot-env/lib/python3.12/site-packages/torchcodec/libtorchcodec_core4.so')"
# → OSError: undefined symbol: torch_dtype_float4_e2m1fn_x2
```

---

## 附录 A：本报告用到的实测命令与原始输出（摘录）

```
# 1) torchcodec 需要的 FFmpeg
ldd .../torchcodec/libtorchcodec_core4.so
  → 需要 libavcodec.so.58 libavdevice.so.58 libavfilter.so.7 libavformat.so.58 libavutil.so.56

# 2) 系统 FFmpeg
ffmpeg -version → ffmpeg version 6.1.1-3ubuntu5
ls /usr/lib/x86_64-linux-gnu/libavcodec.so.* → libavcodec.so.60 libavcodec.so.60.31.102

# 3) 沙箱补齐后仍失败
OSError: .../libtorchcodec_core4.so: undefined symbol: torch_dtype_float4_e2m1fn_x2

# 4) pyav 解码速度（生产路径 decode_video_frames_torchvision）
取 25 帧: 22 ms ⇒ 0.88 ms/帧
跨文件 30 次×10 帧: 462 ms ⇒ 1.54 ms/帧（含每次 open 开销）
```

## 附录 B：沙箱清单（本次新增，均在 /tmp，可随时删除）

| 路径 | 内容 |
|---|---|
| `/tmp/tc_sandbox/libs/` | 41 个 Jammy deb 解出的 89 个 .so（FFmpeg 5 + 依赖） |
| `/tmp/tc_sandbox_libs.tgz` | 上述库的打包（36 MB） |
| `/tmp/ffmpeg5_libs.tgz` | 最初的 7 个 FFmpeg 5 主库 |

**未做任何系统级改动**：未 `apt install`、未改 `/usr/lib`、未改训练 env、未重启服务。
