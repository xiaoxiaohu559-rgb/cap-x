#!/bin/bash
# CaP-X macOS ARM64 一键启动脚本
#
# 启动所有服务并打开 Web UI:
#   bash scripts/start_macos.sh
#
# 仅启动后端服务（无 Web UI）:
#   bash scripts/start_macos.sh --headless
#
# 使用 DeepSeek 模型:
#   bash scripts/start_macos.sh --model deepseek/deepseek-chat
#
# 停止所有服务:
#   bash scripts/start_macos.sh --stop
set -e

cd "$(git rev-parse --show-toplevel)"

# ── 默认参数 ──────────────────────────────────────────────────────────────
MODEL="${MODEL:-anthropic/claude-sonnet-4-6}"
CONFIG="${CONFIG:-env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml}"
SERVER_URL="http://127.0.0.1:8110/chat/completions"
WEBUI_PORT=8200
HEADLESS=false
STOP=false

for arg in "$@"; do
    case "$arg" in
        --headless) HEADLESS=true ;;
        --stop)     STOP=true ;;
        --model)    shift; MODEL="$1" ;;
        --model=*)  MODEL="${arg#*=}" ;;
        --config)   shift; CONFIG="$1" ;;
        --config=*) CONFIG="${arg#*=}" ;;
    esac
    shift 2>/dev/null || true
done

# ── 停止所有服务 ──────────────────────────────────────────────────────────
stop_all() {
    echo "Stopping CaP-X services..."
    for port in 8110 8116 8200; do
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill $pid 2>/dev/null && echo "  Killed PID $pid (port $port)" || true
        fi
    done
    echo "Done."
}

if [ "$STOP" = true ]; then
    stop_all
    exit 0
fi

# ── 环境检查 ──────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Run 'uv venv -p 3.10 && uv sync --extra robosuite' first."
    exit 1
fi

export NO_PROXY=127.0.0.1,localhost
export MUJOCO_GL=glfw
mkdir -p logs

source .venv/bin/activate

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          CaP-X macOS Privileged Mode Launcher           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Model:  $MODEL"
echo "║  Config: $CONFIG"
echo "║  Mode:   $([ "$HEADLESS" = true ] && echo 'Headless CLI' || echo 'Web UI')"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. PyRoKi IK Server (port 8116) ──────────────────────────────────────
if NO_PROXY=127.0.0.1 curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8116/ 2>/dev/null; then
    echo "[✓] PyRoKi IK server already running on :8116"
else
    echo "[~] Starting PyRoKi IK server on :8116..."
    nohup python capx/serving/launch_pyroki_server.py > logs/pyroki.log 2>&1 &
    echo "    PID: $!"
fi

# ── 2. Multi-Provider LLM Proxy (port 8110) ──────────────────────────────
if NO_PROXY=127.0.0.1 curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8110/health 2>/dev/null; then
    echo "[✓] LLM proxy already running on :8110"
else
    echo "[~] Starting LLM proxy on :8110..."
    nohup python capx/serving/multi_provider_server.py --port 8110 > logs/llm_proxy.log 2>&1 &
    echo "    PID: $!"
fi

# ── 等待服务就绪 ──────────────────────────────────────────────────────────
echo ""
echo "Waiting for services..."
for i in $(seq 1 15); do
    ik_ok=$(NO_PROXY=127.0.0.1 curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 1 http://127.0.0.1:8116/ 2>/dev/null || echo "000")
    llm_ok=$(NO_PROXY=127.0.0.1 curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 1 http://127.0.0.1:8110/health 2>/dev/null || echo "000")
    if [ "$ik_ok" != "000" ] && [ "$llm_ok" != "000" ]; then
        echo "[✓] All services ready!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "[!] Warning: some services may not be ready. Check logs/ for details."
    fi
    sleep 2
done

# ── 3. 显示可用 provider ──────────────────────────────────────────────────
echo ""
providers=$(NO_PROXY=127.0.0.1 curl -s http://127.0.0.1:8110/health 2>/dev/null || echo '{}')
echo "LLM Proxy: $providers"
echo ""

# ── 4. 构建前端（Web UI 模式）────────────────────────────────────────────
if [ "$HEADLESS" = false ]; then
    if [ ! -f "web-ui/dist/index.html" ]; then
        echo "[~] Building frontend (first time only)..."
        if [ ! -d "$HOME/.capx_nodeenv" ]; then
            uv pip install nodeenv
            nodeenv ~/.capx_nodeenv --prebuilt --node=20.18.1
        fi
        export PATH="$HOME/.capx_nodeenv/bin:$PATH"
        cd web-ui
        npm install --registry https://registry.npmmirror.com
        npm run build
        cd ..
        echo "[✓] Frontend built"
    fi
fi

# ── 5. 启动 ──────────────────────────────────────────────────────────────
if [ "$HEADLESS" = true ]; then
    echo "Starting headless trial..."
    echo "  python capx/envs/launch.py --config-path $CONFIG --model $MODEL --server-url $SERVER_URL --total-trials 1 --num-workers 1"
    echo ""
    python capx/envs/launch.py \
        --config-path "$CONFIG" \
        --model "$MODEL" \
        --server-url "$SERVER_URL" \
        --total-trials 1 --num-workers 1
else
    echo "Starting Web UI on http://localhost:$WEBUI_PORT"
    echo "  Press Ctrl+C to stop"
    echo ""
    python capx/envs/launch.py \
        --config-path "$CONFIG" \
        --web-ui True \
        --model "$MODEL" \
        --server-url "$SERVER_URL"
fi
