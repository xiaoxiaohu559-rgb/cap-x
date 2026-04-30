# CaP-X macOS ARM64 Privileged Mode 适配指南

## 背景

CaP-X 原生设计为 Linux x86_64 + CUDA 环境。本次适配使其能在 **macOS ARM64 (Apple Silicon)** 上运行 **Robosuite Privileged 模式**——使用仿真器 ground-truth 感知（跳过 SAM3/ContactGraspNet 等 GPU 感知模型），通过 LLM API 生成代码，PyRoKi (JAX/CPU) 求解逆运动学。

---

## 架构概览

Privileged 模式的代码路径：

```
launch.py → runner.py → trial.py
  ├─ 环境: robosuite_cubes.py → robosuite_base.py (MuJoCo, CPU)
  ├─ 任务: franka_pick_place.py → tasks/base.py
  ├─ API:  control_privileged.py → common.py + motion/pyroki.py
  ├─ IK:   PyRoKi Server (JAX, CPU)
  └─ LLM:  llm/client.py (HTTP → OpenRouter/OpenAI API)
```

**不需要的组件**：SAM3, ContactGraspNet, torch/torchvision, CUDA, vLLM, flash-attn, open3d

---

## 改动清单

### 1. `pyproject.toml` — 平台与依赖

| 改动点 | 说明 |
|--------|------|
| `[tool.uv] environments` | 添加 `sys_platform == 'darwin' and platform_machine == 'arm64'` |
| 基础 `dependencies` | Linux-only 包 (torch, torchvision, sam3, open3d, ray, transformers 等) 加 `; sys_platform == 'linux'` marker |
| `override-dependencies` | numpy pin 限制在 Linux；macOS 使用 `numpy>=1.26`  |
| Linux-only extras (verl, molmo, curobo, contactgraspnet, libero) | 所有子依赖加 `; sys_platform == 'linux'` marker |
| `[tool.uv.sources]` | Linux-only source paths 加 `marker = "sys_platform == 'linux'"` |

### 2. MUJOCO_GL 渲染后端 (5 个文件)

macOS 不支持 EGL，改为根据平台自动选择：

```python
import sys
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
```

涉及文件：
- `capx/envs/launch.py`
- `capx/envs/simulators/robosuite_base.py`
- `capx/envs/simulators/robosuite_two_arm_lift.py`
- `capx/envs/simulators/robosuite_nut_assembly.py`
- `capx/envs/simulators/robosuite_handover.py`

### 3. Lazy Import — open3d

`open3d` 在 macOS 上未安装，但 privileged API 实际不使用它。改为 try/except：

```python
try:
    import open3d as o3d
except ImportError:
    o3d = None
```

涉及文件：
- `capx/integrations/franka/common.py`
- `capx/integrations/franka/spill_wipe_privileged.py`

### 4. `capx/integrations/__init__.py` — 条件注册 API

非 privileged API 依赖 open3d 等 Linux-only 包，用 try/except 包裹。macOS 上只注册 6 个 privileged API：

- `FrankaControlPrivilegedApi`
- `FrankaControlNutAssemblyPrivilegedApi`
- `FrankaControlSpillWipePrivilegedApi`
- `FrankaHandoverPrivilegedApi`
- `FrankaTwoArmLiftPrivilegedApi`
- `FrankaControlMultiPrivilegedApi`

---

## 安装步骤

```bash
cd ~/cap-x

# 1. 初始化 robosuite 子模块（其他子模块不需要）
git submodule update --init capx/third_party/robosuite

# 2. 为未初始化的子模块创建 stub（uv 解析需要）
# 已在仓库中创建，跳过即可

# 3. 创建虚拟环境并安装
uv python install 3.10
uv venv -p 3.10
source .venv/bin/activate
uv sync --extra robosuite
```

安装完成后验证：

```bash
MUJOCO_GL=glfw python -c "
import robosuite; print(f'robosuite {robosuite.__version__}')
import mujoco; print(f'mujoco {mujoco.__version__}')
import jax; print(f'jax {jax.__version__}')
import pyroki; print('pyroki OK')
"
```

---

## 运行 Privileged 仿真

### 启动 PyRoKi IK Server

```bash
MUJOCO_GL=glfw python capx/serving/launch_pyroki_server.py
# 默认监听 127.0.0.1:8116
```

### 启动 LLM Proxy（可选，也可直接用 API）

```bash
echo "sk-or-v1-your-key-here" > .openrouterkey
python capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

### 运行评估

```bash
MUJOCO_GL=glfw python capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \
    --total-trials 1 --num-workers 1 \
    --model "openai/gpt-4o" \
    --server-url "http://127.0.0.1:8110/chat/completions"
```

### 运行 Oracle Code 测试（不需要 LLM）

```bash
MUJOCO_GL=glfw python capx/envs/launch.py \
    --config-path env_configs/human_oracle_code/robosuite/franka_robosuite_cube_stack_privileged_oracle.yaml \
    --total-trials 1 --num-workers 1
```

---

## 可用的 Privileged 配置

| 任务 | 配置文件 |
|------|----------|
| Cube Stack | `env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml` |
| Cube Lifting | `env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml` |
| Cube Restack | `env_configs/cube_restack/franka_robosuite_cube_restack_privileged.yaml` |
| Nut Assembly | `env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml` |
| Spill Wipe | `env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml` |
| Two Arm Lift | `env_configs/two_arm_lift/franka_robosuite_two_arm_lift_privileged.yaml` |
| Two Arm Handover | `env_configs/two_arm_handover/two_arm_handover_privileged.yaml` |

Oracle code 版本在 `env_configs/human_oracle_code/robosuite/` 下。

---

## 快速启动（一键脚本）

安装完成后，使用 `scripts/start_macos.sh` 一键启动所有服务：

```bash
# 启动 Web UI（默认，浏览器打开 http://localhost:8200）
bash scripts/start_macos.sh

# 使用 DeepSeek 模型
bash scripts/start_macos.sh --model deepseek/deepseek-chat

# 指定任务配置
bash scripts/start_macos.sh --config env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml

# 无头模式（CLI 直接运行）
bash scripts/start_macos.sh --headless

# 停止所有服务
bash scripts/start_macos.sh --stop
```

脚本会自动：
1. 启动 PyRoKi IK Server（端口 8116）
2. 启动 LLM Proxy（端口 8110，支持 `anthropic/*` 和 `deepseek/*`）
3. 构建前端（首次运行自动安装 Node.js 20）
4. 启动 Web UI（端口 8200）或无头试验

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `anthropic/claude-sonnet-4-6` | LLM 模型名 |
| `CONFIG` | `franka_robosuite_cube_stack_privileged.yaml` | 任务配置文件 |

### API Key 配置

将 API Key 放在项目根目录的文件中：

```bash
echo "your-anthropic-key" > .anthropickey
echo "your-deepseek-key"  > .deepseekkey
```

### Web UI 功能

- 在页面下拉框选择任务配置
- 设置模型名称和 Server URL
- 输入自然语言指令执行仿真
- 实时查看 LLM 生成的代码和执行结果
- 3D Viser 可视化 + 视频回放
- 支持多轮对话，每轮可审查/注入提示

---

## 验证结果

在 macOS ARM64 (Darwin 25.3.0, Apple Silicon) 上测试通过：

- **依赖安装**: 137 个包成功安装，包括 robosuite 1.5.1, mujoco 3.8.0, jax 0.4.29, pyroki
- **核心 import**: numpy, robosuite, mujoco, gymnasium, jax, pyroki, openai, viser 全部通过
- **Privileged API**: 6 个 privileged API 成功注册
- **环境创建**: `FrankaPickPlaceCodeEnv` 成功创建并 reset
- **仿真**: 初始 reward 计算成功 (0.000248)

---

## 限制

- 仅支持 **Robosuite** 仿真器的 **Privileged 模式**
- BEHAVIOR (Isaac Sim) 和 LIBERO 不支持 macOS
- 非 privileged 模式（需要 SAM3/ContactGraspNet 感知）不支持
- RL 训练 (CaP-RL) 不支持（需要 CUDA）
- 多 worker 并行可能受 macOS `spawn` 限制，建议 `--num-workers 1` 开始测试
