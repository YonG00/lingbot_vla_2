# RoboTwin Blackwell 评测去噪运行库

SAPIEN 3.0.0b1 自带 OIDN 2.0.1。在本次 RTX PRO 6000 Blackwell 上，CUDA插件已加载但设备枚举为0，渲染持续输出 `unsupported device type: CUDA` / `invalid handle`，实际策略输入带有明显颗粒。隔离 OIDN 2.5.1 能识别显卡，同seed三视角渲染不再出现OIDN错误，图像颗粒消失。

官方依据：[OIDN支持设备](https://github.com/RenderKit/oidn#supported-devices)、[固定2.5.1发布](https://github.com/RenderKit/oidn/releases/tag/v2.5.1)。只替换进程使用的去噪运行库，不升级SAPIEN、物理、相机或改变32spp。图像改善不代表策略抓取必然成功，必须重跑闭环。

## 安装与启用

在**实验仓库根目录**安装，二进制在忽略目录 `.runtime/`，不修改Conda或生产代码：

```bash
mkdir -p .runtime/oidn-2.5.1
curl -fL --retry 2 -o .runtime/oidn-2.5.1/release.tar.gz \
  https://github.com/RenderKit/oidn/releases/download/v2.5.1/oidn-2.5.1.x86_64.linux.tar.gz
echo '743c3e2aff8c220d5d70fe6cb970fb3d36f2702d2693c61d1d148e404cf37cd6  .runtime/oidn-2.5.1/release.tar.gz' | sha256sum -c -
tar -xzf .runtime/oidn-2.5.1/release.tar.gz -C .runtime/oidn-2.5.1
```

服务器无法下载时，在本机下载同一个官方包、校验后scp，再解压。不要下载同名其他版本替代。

`RoboTwin-lingbot/envs/__init__.py` 在导入任何SAPIEN模块前调用 `render_runtime.configure_oidn()`。发现上述目录自动使用；未安装且未显式设置时保持原自带后端。也可在**仿真进程启动前**设置 `ROBOTWIN_OIDN_LIBRARY_DIR=<解压后的lib绝对路径>`。指定目录缺失、版本文件不全、导入过晚或没有CUDA设备时立即失败，避免把有噪声的结果当正常评测。

启动日志必须出现 `[render_runtime] OIDN 2.5.1 CUDA=[...]`。该标记只证明枚举通过；还需确认真实相机渲染完成、无 `OIDN Error`，并检查 `/proc/<sim_pid>/maps` 中只有2.5.1的core/base/CUDA插件。旧版绝对路径预载shim被进程局部替代，site-packages保持原样。

回滚：新仿真进程设置 `ROBOTWIN_OIDN_LIBRARY_DIR=bundled`；旧缺陷也会恢复。环境变量读取一次，必须重启仿真。已有运行进程不会改变。

## 本次验证

- CPU加载契约：`PYTHONPATH=.:tests python -m pytest -q tests/mouse_pad_training/test_render_runtime.py`，覆盖显式回滚、无效目录、过晚导入、CPU/SYCL不能冒充CUDA及有效CUDA路径。
- 隔离对照：seed100000、鼠标model2、位置完全相同，三视角各3帧；旧版加载2.0.1并报错，新版只加载2.5.1，渲染无OIDN错误，视觉颗粒消失。
- 正式接入静态渲染通过；实际闭环进程maps仅2.5.1，三路真实策略输入颗粒消失，初始机器人状态与旧场景一致。固定新step500、seed100000、noise0、chunk50完整400步/8次forward，OIDN错误0，但成功率0/1、鼠标最大抬升0.0724mm；说明渲染修复不能单独解决此次抓取失败，不能将1回合推论为总体成功率。远端结果：`/data/outputs/mouse_render_fixed_20261019_v1`。
- CPU加载契约本地/服务器7 passed；本地配方与启动续训回归62 passed、1 skipped（缺torch装配项）、36 subtests。原用户修改的评测客户端hash前后相同，结束后无残留GPU进程，机器保持开机。

### 修复后10步动作块补测

用户要求补齐10步档，实际运行 `use_length=10`；模型内部去噪迭代仍为10，不修改模型、相机、精度、场景或推理噪声种子。两档均完成400动作，实际初始三视角和状态逐位相同、指令一致。

| 每次执行动作数 | 实际推理次数 | 成功 | 鼠标最大抬升 | 最终水平位移 | OIDN错误 |
|---|---:|---:|---:|---:|---:|
| 50 | 8 | 0/1 | 0.0724mm | 125.85mm | 0 |
| 10 | 40 | 0/1 | 0.1653mm | 22.04mm | 0 |

`sample_actions` 日志稳态中位数（去掉首次）分别0.5450s、0.5399s；累计5.257s、22.628s，反映10档调用次数更多。首份forward记录到最后动作落盘的区间分别72.28s、80.71s，不含模型加载、专家预检及首次推理，不能作为整个评测总耗时。

两档都没有稳定抓起鼠标。10档水平位移较小不等于成功率提高，只是同一回合观察。相同输入/噪声seed下首块预测也未逐位相同（归一化前10帧最大差0.05859，原始关节动作最大差0.01442），尚未隔离其原因；不把轨迹差异全部归因于动作块长度。补测结果在 `/data/outputs/mouse_render_fixed_len10_20261019_v1`，生成摘要/对比在本地 `artifacts/data_audits/mouse/20261019_render_fix/len10/`。两次均验证原客户端hash不变，结束后无GPU进程。

本次诊断生成的JSON、NPZ、PNG、进程maps和日志放在实验 `.runtime/render_check/` 及 `artifacts/data_audits/mouse/`，不放docs、不进入Git。
