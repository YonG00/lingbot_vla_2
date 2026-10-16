# RoboTwin 评测全景录像（外部观察相机）

> 2026-10-16：已本地实现，22 项 CPU 测试通过，含真实 H.264 编码；未同步服务器，GPU 构图/黑帧/性能待验。
> 目标：人工观察桌面、双臂、操作物体及目标区域。相机只用于录像，不作为模型输入，不涉及 AL 训练。

## 使用方法

不需要手动启动第二个录屏程序。现有测评自动调用独立录像模块，每回合保存全景 MP4。

| 需求 | 两阶段启动器追加参数 | benchmark 追加参数 |
|---|---|---|
| 只录全景，默认 60 FPS | --global_view | --global-view |
| 全景 30 FPS | --global_view --global_view_fps 30 | --global-view --global-view-fps 30 |
| 同时录 head | --enable_video --global_view | --video --global-view |
| 都不录（默认） | 不加录像参数 | 不加录像参数 |

单任务 clean 示例（Python 包根目录执行，替换 HF 路径）：

```bash
MODEL='/data/outputs/<RUN>/checkpoints/global_step_500/hf_ckpt'
TASKS=/tmp/global_view_put_object_cabinet.txt
printf '%s\n' put_object_cabinet > "$TASKS"

QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path "$MODEL" --eval_workdir /data/code/RoboTwin-lingbot \
  --conda_sh /data/miniconda3/etc/profile.d/conda.sh \
  --inference_env lingbotvla --sim_env RoboTwin \
  --task_config demo_clean --task_list_file "$TASKS" --test_num 1 \
  --num_gpus 1 --num_per_gpu 1 \
  --use_bf16 True --use_fp32 False --use_compile False \
  --global_view --global_view_fps 60 \
  --output_base /data/eval_results/global_view
```

前提：环境和模型已准备好，ACT 配置 instruction_type 已设为需要的 unseen；命令不修改它。
服务器长任务按仓库约定使用 setsid/nohup 后台并重定向日志。本例不包含关机/清理。

## 产物

启动器输出目录的 `eval_results/<task>/` 下新增：

- `episode0_global.mp4`：1280×720、H.264 视频。
- `episode0_global.preview.png`：初始画面，回合中即可检查构图。
- `episode0_global.json`：结束时记录相机/FPS/帧数/物理步数/仿真和视频时长/result/encoder_ok。
- `episode0_global.ffmpeg.log`：编码错误日志，正常可为空。

全景文件名固定；成功/失败写在 JSON 的 result 中。head 仍按原 success/failure 命名，画面与原帧率不变。
benchmark 录像位于其输出目录的 `launcher_outputs/<config>/<launcher_run>/eval_results/<task>/`。
正常结束和 Python 异常会关闭编码器；异常标记 interrupted。强杀/掉电不能保证 MP4 完整收尾。
benchmark 仍按完整任务续测；全景开关/FPS 改变需用新 run-name，不会给旧任务自动追补录像。

## 60 FPS 的含义和代价

默认物理步长 1/250 秒，每隔 4～5 个物理子步采一个新画面，实现平均 60 FPS，不复制旧图、不加速播放。
视频时间是仿真时间，不包含模型推理/加载等待；采样时钟跨动作及 action chunk 连续。
保留初始帧；末状态落在采样间隔中时补一帧，视频时长与仿真时长差在两个帧间隔以内。
支持 1～120 FPS，且不得高于物理步频；原 head 的 --video_fps 仅改 head 播放帧率，不控制全景采样。
60 FPS 足够流畅，相同仿真时长的采帧量是 30 FPS 的两倍；额外渲染、图像回传、CPU 编码会增加耗时，墙钟增量待真机实测。编码为 veryfast/CRF23/两个 CPU 线程。

## 构图

独立外部斜俯视，默认 position=[1.2,-1.2,1.8]、target=[0,0,0.85]、fovy=65°、1280×720。
尚未 GPU 验证，不保证所有任务无遮挡；先看 preview PNG。可在 task_config YAML 顶层调整（仅全景开启时生效，下次启动读取）：

```yaml
global_view_camera:
  position: [1.2, -1.2, 1.8]
  target: [0.0, 0.0, 0.85]
  fovy: 65
```

固定视角仍可能被柜子/手臂遮挡。需要观看时自由旋转/缩放，使用 SAPIEN Viewer + 远程桌面；MP4 不提供交互视角。

## 实现与同步

| 文件 | 职责 |
|---|---|
| RoboTwin-lingbot/envs/utils/global_view_video.py | 独立录像模块：时钟、ffmpeg、末帧、预览回调、JSON |
| RoboTwin-lingbot/envs/camera/camera.py | create_global_view_camera；确定性相机，不加入策略相机列表 |
| RoboTwin-lingbot/envs/_base_task.py | 生命周期及 take_action 内物理子步采帧 |
| lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py | 实际客户端，回合录像和异常清理 |
| lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh | 参数透传，模型启动前检查 RoboTwin 是否含新功能 |
| lingbot-vla-v2/experiment/robotwin/run_clean_benchmark.py | benchmark 参数和续测 manifest 身份 |

启动器会把实验包内客户端复制到 RoboTwin/script，不能只改副本；无需改官方 script/eval_policy.py。
相机/基类/新录像模块不会自动覆盖，必须经获批 Git 同步一起更新到实际 RoboTwin clone；缺失时开启全景会在启动模型前拒绝运行。
不复用 observer_camera 和 world_camera1/2。world_camera2 旧重复名称记作已知问题，本功能不使用，暂不改。

## 验证与回退

Python 包根目录：

```bash
PYTHONPATH=.:tests python3 -m unittest \
  test_robotwin_global_view_video test_clean_benchmark_nvidia_adapter -v
```

22 项 CPU 通过：物理时钟 1/30/60/120 FPS、末帧、相机正交基、编码异常/超时、真实 H.264 60 FPS/61帧/1.016667秒、head×全景四种组合、模型输入一致、推理/启动异常清理、参数/续测身份。语法/bash -n/diff 检查通过。
GPU 待验：clean 单任务 1 回合的构图、非黑帧、运动连续性、实际帧数/时长、关闭/30/60 FPS 耗时对比；禁止和训练同时测。
回退：下次启动去掉 --global_view / --global-view，即不新增相机/采帧/ffmpeg；无需删除录像或改训练配置。
