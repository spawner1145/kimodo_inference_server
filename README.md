# Kimodo 推理服务

独立 HTTP 服务，支持 Kimodo 的 SOMA、SMPL-X、G1 动作模型。Blender 不加载 PyTorch 或模型，只连接 API、提交任务并下载动画。服务端与插件可放在不同电脑上。模型文件通过配置引用，不复制到代码目录。

## 安装环境

推荐 **64 位 Python 3.12**，最低 Python 3.10。Python 可从 [官网](https://www.python.org/downloads/) 安装；GPU 推理需要 NVIDIA GPU 和兼容的显卡驱动。以下命令均在 `inference_server` 目录执行。

**支持全局 Python 和虚拟环境，不强制创建或激活虚拟环境。** 两种方式安装相同依赖、使用相同配置。全局安装与该 Python 的其他项目共享依赖，虚拟环境则单独存放依赖；下面第 1 步任选一种，然后继续第 2–5 步。

### 1. 选择 Python 环境

#### 方式 A：直接使用全局 Python

无需运行 `python -m venv`。如果终端正在使用虚拟环境，先执行 `deactivate` 退出，或直接指定全局 Python 的完整路径。

Windows PowerShell（选择已安装的 Python 3.12）：

```powershell
cd D:\code\kimodo\inference_server
$env:KIMODO_PYTHON = (py -3.12 -c "import sys; print(sys.executable)").Trim()
& $env:KIMODO_PYTHON -m pip install --upgrade pip setuptools wheel
```

如果没有 `py` 启动器，直接设置实际路径，例如 `$env:KIMODO_PYTHON='C:\Python312\python.exe'`。后续教程里的每条 `python ...` 都可以改成 `& $env:KIMODO_PYTHON ...`，确保 PyTorch、requirements、Kimodo 和服务端安装到同一个全局 Python。安装完成后在同一终端执行 `.\start.bat`，它会优先使用这个解释器，即使目录旁还有 `.venv`。

Windows CMD 对应写法为 `set "KIMODO_PYTHON=C:\Python312\python.exe"`，随后使用 `"%KIMODO_PYTHON%" -m pip ...` 和 `start.bat`。

Linux / macOS（先退出已激活的虚拟环境）：

```sh
cd /path/to/inference_server
export KIMODO_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
"$KIMODO_PYTHON" -m pip install --upgrade pip setuptools wheel
```

后续把 `python ...` 改成 `"$KIMODO_PYTHON" ...`，安装完成后在同一终端执行 `sh start.sh`。也可以直接指定解释器，例如 `export KIMODO_PYTHON=/usr/local/bin/python3`，以实际安装位置为准。如果发行版的系统 Python 提示 `externally-managed-environment`，请使用自行安装的 Python 或下面的虚拟环境方式。

`KIMODO_PYTHON` 在上述示例中只对当前终端及其子进程生效；新开终端或直接双击启动脚本时，需要重新设置或将它保存为用户环境变量。它可以指向全局 Python，也可以指向任意虚拟环境的 Python，值只包含解释器路径。

#### 方式 B：创建并使用虚拟环境（可选）

Windows PowerShell：

```powershell
cd D:\code\kimodo\inference_server
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

如果使用 Windows CMD，激活命令是 `.venv\Scripts\activate.bat`。也可以不激活，把后续命令里的 `python` 换成 `.\.venv\Scripts\python.exe`，这样无需调整 PowerShell 执行策略。已有合适的虚拟环境时直接使用，无需再建一个；本机现有环境在相邻的 `.venv-nvfp4`。

Linux / macOS：

```sh
cd /path/to/inference_server
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Linux 如果提示缺少 `venv` 模块，先通过发行版包管理器安装对应的 `python3-venv` 包，再创建环境。

### 2. 手动安装 PyTorch

**先装 PyTorch，再装 requirements**，以便明确选择 CUDA 或 CPU 构建。按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择适合操作系统和 NVIDIA 驱动的 CUDA 版本。以下是 Windows / Linux 的 CUDA 12.8 示例，版本只设下限：

```sh
python -m pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128
```

本服务不需要安装 `torchvision` 或 `torchaudio`。PyTorch 官方 wheel 带有 CUDA 运行库，普通推理不要求另装完整 CUDA Toolkit，关键是显卡驱动兼容。`nvidia-smi` 中的 CUDA 版本表示驱动支持能力，不等同于当前 Python 环境的 PyTorch 构建。

安装后检查：

```sh
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

NVIDIA GPU 推理应显示 `GPU available: True`。如果是 `False`，先检查当前使用的 Python、驱动以及是否装成 CPU wheel；升级驱动后重新打开终端再试。

只使用 CPU 时，Windows / Linux 可改用：

```sh
python -m pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cpu
```

macOS 使用官方安装页给出的 macOS 命令（通常为 `python -m pip install "torch>=2.8"`）。CPU / macOS 运行时在 `config.json` 中设置 `device: "cpu"`，选择 FP32 动作模型和支持 CPU 的编码器；当前 ConvRot INT8 CUDA 路径不可用，完整大文本编码器也需要足够内存。当前服务没有 MPS 推理适配。

### 3. 安装推理依赖

```sh
python -m pip install -r requirements.txt
```

`requirements.txt` 中的直接依赖**只设最低版本，不锁定版本或设置上限**，包括 `torch>=2.8`、`transformers>=5.1.0`、PEFT、Accelerate、WebUI/API 和导出依赖。它不指定 PyTorch 下载源；已经按上一步装好的 PyTorch 满足要求时，普通安装不会主动升级它。请勿在无意更换 CUDA 构建时对全部依赖使用 `--upgrade`。

Windows / Linux 默认包含 NF4 所需的 bitsandbytes，macOS 通过平台条件跳过。NF4 完整模型仍需后续下载；安装依赖不下载模型权重。依赖版本下限不代表所有未来版本都已验证兼容；本机使用 Transformers 5.1.0 完成过验证，升级后如遇上游内部 API 变化，需要相应适配。

### 4. 安装 Kimodo 核心和本服务

本文件只列推理所需依赖，Kimodo 核心代码需要单独安装。上游完整依赖含演示界面和自身的固定版本声明，下面使用 `--no-deps`，由本项目的 requirements 管理推理依赖；`SKIP_MOTION_CORRECTION_IN_SETUP=1` 跳过可选 C++ 后处理编译。

如果服务端目录旁就是官方 Kimodo 源码（本机当前目录结构即如此），在 Windows PowerShell 执行：

```powershell
$env:SKIP_MOTION_CORRECTION_IN_SETUP='1'
python -m pip install --no-deps --no-build-isolation -e ..
python -m pip install --no-deps --no-build-isolation -e .
```

Linux / macOS：

```sh
export SKIP_MOTION_CORRECTION_IN_SETUP=1
python -m pip install --no-deps --no-build-isolation -e ..
python -m pip install --no-deps --no-build-isolation -e .
```

**如果只拿到了独立的服务端文件夹**，父目录不是 Kimodo 源码，把上面的 `-e ..` 安装命令换成以下命令（同样先设置 `SKIP_MOTION_CORRECTION_IN_SETUP=1`）：

```sh
python -m pip install --no-deps --no-build-isolation "https://github.com/nv-tlabs/kimodo/archive/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954.zip"
python -m pip install --no-deps --no-build-isolation -e .
```

这是 NVIDIA 官方仓库的源码提交，与库版本上限无关，不需要安装 Git。网络无法访问 GitHub 时，也可从该仓库下载并解压源码，再把安装命令的 URL 替换为解压后的本地目录。跳过后处理时不需要 CMake 或 C++ 编译器；想使用 `motion_correction` 的用户按 Kimodo 官方文档另行编译安装。

检查基础推理和文本编码器代码能否加载：

```sh
python -c "import torch, kimodo, kimodo_server; from kimodo_server.runtime import MotionRuntime; from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel; print('Imports OK; CUDA:', torch.cuda.is_available())"
```

### 5. 配置模型并启动

仓库提供 `config.example.json`；第一次启动时，若同目录没有 `config.json`，会自动复制示例生成本地配置，已有配置不会覆盖。默认配置使用服务端目录内的 `models/` 存储全部动作模型、文本编码器及其适配器，即 `kimodo_inference_server/models/`，运行数据保存在 `data`。第一次使用时可启动服务，在 WebUI 的模型管理中下载权重，或编辑配置指向已有权重；缺少权重不影响打开面板，生成前才需要相应模型和编码器齐全。模型和编码器路径按 `models_dir` 解析。

默认模型目录结构（下载后创建对应子目录）：

```text
kimodo_inference_server/
├─ models/
│  ├─ Kimodo-SOMA-RP-v1.2/               FP32 / FP16 / BF16 共用原始权重
│  ├─ Kimodo-SOMA-RP-v1.2-int8-convrot/  INT8 权重
│  └─ text_encoders/                    AWQ / NF4 / BF16 编码器及适配器
├─ data/                               运行缓存与生成结果
└─ config.json
```

WebUI 和命令行下载均使用同一个 `models_dir`，模型加载也从这里读取；相对路径以配置文件所在目录为基准。`models/` 已被 `.gitignore` 排除。旧安装如曾使用 `../models`，需将对应权重子目录放到新的 `models/` 下；修改配置不会自动搬动权重。

Windows 双击 **`start.bat`**；Linux / macOS 执行 **`sh start.sh`**。启动后打开 **http://127.0.0.1:6600/**，API 文档在 `/docs`，按 Ctrl+C 停止。启动脚本不安装依赖、不自动下载模型；已有服务运行时无需重复启动。

两个脚本都从自身目录读取 `config.json`，不受工作目录影响；按顺序选择 `KIMODO_PYTHON` 指定的解释器、已激活虚拟环境、服务端目录的 `.venv`、相邻 `.venv-nvfp4`、相邻 `.venv`，最后使用 PATH 中的 Python（Windows 为 `python`，Linux / macOS 优先 `python3`）。因此没有虚拟环境时，也能直接使用全局 Python；希望忽略旁边已有的虚拟环境时，设置 `KIMODO_PYTHON` 为全局解释器的完整路径。脚本会显示实际使用的 Python。Linux 也可先 `chmod +x start.sh` 再执行 `./start.sh`。

支持追加参数，例如 `start.bat --port 6601` 或 `sh start.sh --port 6601`。自定义解释器时，Windows PowerShell 使用 `$env:KIMODO_PYTHON='C:\path\to\python.exe'`，Linux 使用 `KIMODO_PYTHON=/path/to/python sh start.sh`。只填写解释器路径，不附带参数。Windows 启动失败会保留窗口显示错误，自动化调用可设置 `KIMODO_NO_PAUSE=1`。

也可以直接使用已安装好依赖的全局 Python 或虚拟环境 Python 手动启动，无需通过启动脚本：

```sh
python -m kimodo_server serve --config config.json
```

## 中文 Web 管理面板

启动后打开 **http://127.0.0.1:6600/**，无需单独安装前端或连接外部 CDN。

- **模型管理**：查看本地就绪状态、设置默认动作模型/编码器，编辑下载源（Hugging Face / ModelScope）、仓库 ID、权重目录、精度、骨架；支持新增和移除配置。移除配置不删除模型文件。
- **下载任务**：在模型卡片上点击“下载”。下载源由对应配置的 `download_source` 决定。HF 下载可指定兼容镜像端点和 HF 令牌；ModelScope 使用官方站点及独立的 ModelScope 令牌，公开仓库无需登录。可查看进度、失败原因，取消后再次下载可续传。下载子进程单独联网，推理继续使用本地离线权重。完整 NF4/BF16 不会自动下载。
- **推理配置**：步数、CFG、随机种子、过渡帧、输出格式、动作 block swap、CPU 卸载、TF32 和 CPU 线程；文本 block swap 在对应编码器的编辑框中设置。这里的采样默认值仅用于请求未提供的字段，Blender 显式发送的参数优先。
- **服务设置**：监听地址、默认 **6600** 端口、模型根目录、运行数据目录、队列与保留时间。
- 修改后先“保存配置”，再“应用并重启”。配置原子写入启动时指定的 JSON 文件，页面区分待应用与已生效；有推理或下载任务时拒绝重启。重启前检查新监听地址是否可用，端口变化后页面自动跳转，Blender 需使用新地址重新连接。

下载期间不允许加载/生成、修改配置或重启，避免下载和推理同时读写权重。HF 下载进度按仓库文件计数；ModelScope 面板按仓库完成数更新，下载期间显示当前仓库，命令行下载时可显示 SDK 的传输进度；均不是面板中的精确字节进度；任务历史在重启后清空，续传数据保留。取消或失败留下的 `.kimodo-downloading` 标记会使该权重目录保持“未就绪”，成功重新下载后清除。分片编码器需所有索引权重文件存在才标记可用。

启用 `KIMODO_API_TOKEN` 后，在面板左下角填写相同 API 令牌。令牌只保留在页面内存中；刷新或跳到新端口需重新填写。下载令牌不写配置文件或任务信息，可留空使用服务进程中对应的 `HF_TOKEN` / `MODELSCOPE_API_TOKEN`，两类下载凭据分开使用。对外监听前需在服务进程环境设置 API 令牌并重新启动服务。

迁移到其他目录或机器：在可运行 NVIDIA Kimodo v1.2 的 Python 环境中执行 `pip install -e .`，修改 `config.json` 的 `models_dir`、编码器路径和模型配置。运行时需要 CUDA PyTorch、Kimodo、Transformers、PEFT、Accelerate、bvhio 及它们的依赖；本包不替换已有 CUDA PyTorch。只安装核心 Kimodo、暂不编译可选后处理时，可按上游约定设置 `SKIP_MOTION_CORRECTION_IN_SETUP=1`。当前本机没有 `motion_correction`，默认关闭动作后处理；API 会明确拒绝需要此扩展的请求。

跨机器访问：将 `host` 改为服务端网卡地址或 `0.0.0.0`，先设置 `$env:KIMODO_API_TOKEN='你自己的令牌'`，在 Blender 中填写相同令牌。按需配置系统防火墙；公网使用带 TLS 的反向代理。令牌只从服务进程环境读取，不写入配置文件。

## 配置和优化

- `soma-fp32`：默认、完整精度动作模型。
- `soma-fp16` / `soma-bf16`：复用 `Kimodo-SOMA-RP-v1.2` 的原始浮点权重，在加载时将神经网络权重转换为 FP16/BF16。无需额外下载或保存两份模型；可在 WebUI 的模型管理中设为默认，也可在 Blender 的动作模型下拉框选择。
- `soma-int8`：使用 [spawner/Kimodo-SOMA-RP-v1.2-int8-convrot](https://www.modelscope.cn/models/spawner/Kimodo-SOMA-RP-v1.2-int8-convrot) 的 ConvRot INT8 W8A8 权重，已配置 `download_source: "modelscope"`，可从模型卡片下载，使用真实 INT8→INT32 CUDA GEMM。
- 配置还列出官方注册表的全部 9 个 Kimodo 动作模型：SOMA RP v1/v1.1/v1.2、SOMA RP ArdyRep v1.2、SOMA SEED v1/v1.1、SMPLX RP v1、G1 RP/SEED v1。缺失权重的配置显示“未下载”，不会自动下载；TMR 是动作检索模型，未作为动作生成器暴露。
- 编码器可通过每个请求的 `encoder` 选择：`awq`、`nf4`（原插件的 Aero-Ex 预量化 LLM2Vec）、`bf16`（原始基座与适配器）。NF4 权重已合并适配器，不会再叠加原始 LoRA。
- `offload: true`：文本编码与动作生成分阶段使用显存；编码器留在 CPU 待命，重复提示词可跳过编码。
- 文本特征：CPU LRU + 磁盘缓存，按编码器配置和权重文件身份隔离。内存最多 128 条，磁盘每个编码器最多 4096 条。
- 动作模型常驻、单 GPU 工作线程排队，避免多个请求同时争抢显存。切换配置时替换当前动作模型。
- PyTorch inference mode、显式 QKV/SDPA，FP32 和 INT8 共享同一注意力实现。`tf32` 默认关闭，可在配置中显式开启；开启后数值可能改变。
- 每次请求可配置扩散步数、文本/约束 CFG、分段随机种子、过渡帧、后处理、BVH/NPZ。100 步为基准；减少步数可能降低质量。
- **Block swap**：请求的 `block_swap` 控制动作模型，`text_block_swap` 控制文本编码器。`null` 跟随配置，`0` 关闭，正数为留在 CPU 并逐次交换的 Transformer 块数量；v1.2 动作模型有 32 块，Llama 编码器有 32 层。也可在服务配置顶层及各编码器的 `block_swap` 中设置默认值。此实现采用同步 CPU↔CUDA 交换，兼容 FP32、FP16、BF16、ConvRot INT8 和预量化 NF4；不会改变权重精度，但通常增加延迟，不是加速开关。
- 取消请求会在扩散步之间检查，文本编码或模型加载阶段会在该阶段完成后响应。取消任务不会产生可下载的成功结果。

缓存、生成文件和日志放在 `data/`，不散落到源码或系统临时模型目录。生成结果默认保留 168 小时，由 `retention_hours` 控制。服务重启后任务查询表会重建，缓存仍保留。

### FP16 / BF16 动作推理

模型配置的 `precision` 支持 `fp32`、`fp16`、`bf16`、`int8_convrot`，WebUI 编辑模型时可以选择。FP16/BF16 使用普通 Kimodo 浮点检查点；INT8 必须指向专门量化后的目录，不能仅修改配置就把普通权重变成 INT8。

FP16/BF16 模式把 root/body 两阶段神经网络权重和主要矩阵计算降到相应精度，并在神经网络边界转换回 FP32；扩散采样、阶段之间的根运动转换和最终动作几何计算保持原来的精度，骨架常量和归一化统计不随网络转成半精度。LayerNorm 等算子按 PyTorch autocast 规则处理。这样能够降低权重显存占用，同时避免整套采样器调用 `.half()` 引起数值问题。

两种模式都支持 CPU 卸载和动作 block swap；CUDA BF16 要求 GPU 支持原生 BF16，不支持时明确报错，不会静默回退。CPU 也允许使用半精度配置，性能取决于 PyTorch 和 CPU 支持；GPU 上可优先尝试 BF16，FP16 的数值范围更小。生成发现 NaN/Inf 时任务会失败，不会交付无效动作。

这些选项控制**动作模型**。文本编码器仍由 `encoder` 独立选择 AWQ/NF4/原始 LLM2Vec；选择 `soma-bf16` 不会自动下载或切换成 BF16 Llama。API 示例：

```json
{"model":"soma-bf16","encoder":"awq","prompts":["A person walks forward."],"durations":[2],"steps":100,"seed":42,"block_swap":0}
```

任务结果会记录 `precision`、`neural_weight_dtype`、`sampling_dtype` 和显存峰值。首次增加配置或更新代码后需重启服务；Blender 重新连接 API 后会刷新模型列表。FP16/BF16 可能改变生成动作，不应据此宣称无质量损失。

本机 RTX 4070 Laptop 的一次实际验证：同一缓存的 AWQ 文本特征、提示词 `A person walks forward.`、种子 42、2 秒、100 步。FP32/FP16/BF16 的动作阶段显存峰值约为 1146 / 582 / 582 MiB；FP16、BF16 开启 16 块 swap 后均约为 342 MiB，且与各自不交换的输出逐元素一致。相对 FP32，FP16 的世界坐标/去根 MPJPE 为 3.41/0.50 mm，BF16 为 31.38/3.52 mm（30 个预测关节）。这是单个样本的动作变化与运行检查，不能推广为整体质量结论；文本编码器加载的显存不包含在这些动作阶段数值里。

默认文本编码器为本机的公开 AWQ Llama 3 + 原始 LLM2Vec 两组适配器。它是**近似条件编码器**，不是官方 BF16 LLM2Vec 的等价版本；AWQ 执行为逐层临时反量化参考实现，未宣称融合 AWQ 内核。`encoders.bf16.kind="llm2vec"` 需提供本地原始 Llama 基座、MNTP 和 supervised 权重。`encoders.nf4.kind="nf4_llm2vec"` 使用 `Aero-Ex/KIMODO-Meta3_llm2vec_NF4`，需要 bitsandbytes（可安装 `pip install -e .[nf4]`）。

**验证范围**：本机真实完成 SOMA v1.2 FP32/INT8、AWQ 编码、动作与文本 block swap、BVH 导入、多提示词和取消任务。SOMA/SMPLX/G1 三种骨架的 BVH 导出均用标准骨架检查通过。NF4 使用临时创建的小型预量化 Llama 验证了加载、编码、CPU/GPU 搬移与 block swap 一致性；完整 Aero-Ex NF4、BF16、其他动作模型权重未下载，不能称为完整模型质量已验证。原插件本身将 SMPLX/G1 标为暂不支持；此服务提供它们的通用加载和导出路径。

显式下载已配置权重的命令（仅用户调用下载命令时联网，服务推理保持离线）：

```powershell
python -m kimodo_server download --config config.json --encoder nf4
python -m kimodo_server download --config config.json --model g1-rp-v1
# 从公开 ModelScope 仓库下载 ConvRot INT8，无需 HF 账号
python -m kimodo_server download --config config.json --model soma-int8
```

可用 `--endpoint` 指定兼容 Hugging Face 的镜像，该参数只作用于 HF 配置。`download_source` 未填写时默认 `huggingface`；设为 `modelscope` 时使用 ModelScope 官方 SDK 下载，`repo_id` 填写魔搭仓库 ID。编码器及其适配器使用同一下载源，切换源时需核对所有仓库 ID。需要授权的 HF / ModelScope 仓库分别从 `HF_TOKEN` / `MODELSCOPE_API_TOKEN` 环境变量读取令牌。下载后的 INT8 目录含权重、量化描述、配置与统计数据，仍通过本服务的量化加载器运行。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | 中文 Web 管理面板 |
| GET / PUT | `/v1/admin/config` | 当前/已保存配置、校验并持久保存 |
| POST | `/v1/admin/apply` | 空闲时应用配置并重启 |
| GET | `/v1/admin/status` | 推理与下载任务列表 |
| POST | `/v1/admin/downloads` | 显式下载模型或编码器 |
| POST | `/v1/admin/downloads/{id}/cancel` | 取消下载，保留续传 |
| GET | `/v1/health` | 能力、当前模型、编码器、显存策略 |
| GET | `/v1/models` | 可选择的模型配置 |
| GET | `/v1/encoders` | 文本编码器配置与权重可用性 |
| POST | `/v1/models/load` | 排队加载模型，返回任务 |
| POST | `/v1/jobs` | 提交动作生成 |
| GET | `/v1/jobs/{id}` | 状态、进度、错误、生成参数与耗时 |
| POST | `/v1/jobs/{id}/cancel` | 取消任务 |
| GET | `/v1/jobs/{id}/artifact` | 下载 BVH 或 NPZ 文件 |

请求示例，单段和多段使用相同接口：

```json
{
  "model": "soma-fp32",
  "encoder": "awq",
  "block_swap": 0,
  "text_block_swap": 0,
  "prompts": ["A person walks forward.", "A person waves with the right hand."],
  "durations": [3, 2],
  "seed": 42,
  "seeds": [42, 17],
  "steps": 100,
  "transition_frames": 5,
  "cfg_text": 2,
  "cfg_constraints": 2,
  "post_processing": false,
  "output_format": "bvh",
  "standard_tpose": true,
  "constraints": []
}
```

任务状态：`queued → running → succeeded / failed / cancelled`，取消中为 `cancelling`。提交成功返回 HTTP 202，队列满返回 429，参数错误返回 422。生成接口仅接受已配置的模型 ID；管理接口允许修改配置路径，不接受 Python 代码。

Blender 场景约束编辑器沿用 SOMA 关节映射；SMPLX/G1 可以生成、导入及使用手动/同名骨骼重定向，但不能套用 SOMA 场景约束。API 本身可接收对应骨架的原生约束。非 SOMA 导出 BVH 使用标准静置姿势。

## 量化与动作误差对比

```powershell
# 旋转、整数矩阵乘、注意力 mask 的 CUDA 自检
python -m kimodo_server self-test

# 量化到一个新目录，不覆盖已有模型；之后在 config.json 中添加此路径
python -m kimodo_server quantize --source models/Kimodo-SOMA-RP-v1.2 --output models/my-int8

# 同提示词、种子、采样设置生成完整动作，再比较 30 个实际预测关节
python -m kimodo_server compare --config config.json --prompts "A person walks forward." --seeds 17 42 --duration 4 --steps 100 --output data/comparison.json
```

`compare` 输出世界坐标 MPJPE、去根位置 MPJPE、根轨迹 ADE、局部旋转误差和生成时间；共享相同文本特征，不把层输出误差或文件压缩率当作动作质量。它测量化前后生成动作的变化，不是自然度或相对真实动作的评分。请在 API 空闲或停服时运行量化/对比命令，以免额外占用显存。

ConvRot INT8 为本地适配，作者当前官方实现面向 NVFP4，并未发布 Kimodo INT8 方案。本机 RTX 4070 Laptop 不支持其 NVFP4 路径。此前 24 对动作评测中，INT8 世界坐标 MPJPE 为 213.50 mm、去根位置 MPJPE 为 16.48 mm；这些是特定提示词/种子的观测，不是无损替代或自然度结论。

参考：NVIDIA Kimodo（Apache-2.0）、Aero-Ex/kimodo 的模型常驻/缓存/卸载思路（commit `3b45cc6d420136bfa6128e8be5e81e45521f2b7a`）、ConvRot（commit `c80281c25ad99eb4837abb4317978e12c05d05e1`）。模型权重分别遵循其原始许可。

## 上传 GitHub

将本目录作为独立服务端仓库上传，Blender 插件在另一个文件夹单独上传。`.gitignore` 排除虚拟环境、模型权重、`data/`、缓存、日志、生成动画、打包归档和本机凭据。`config.json` 是 WebUI 可修改的本地配置，不上传；请保留并提交 `config.example.json`、源码、WebUI 静态文件、测试、启动脚本、requirements 和许可文件。首次启动会从示例创建本地配置，不需要安装环境脚本。

需要改变公开默认设置时修改示例文件；本机配置继续在 `config.json` 中修改。模型权重存放在 ModelScope，GitHub 只保存仓库 ID 和加载代码。`.gitignore` 不会取消跟踪以前提交过的文件，若自己的仓库此前已跟踪本地配置或模型，需先从 Git 索引移除并检查历史。
