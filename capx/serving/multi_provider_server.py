"""
Multi-provider LLM proxy server for CaP-X.

Routes requests to the appropriate provider based on model name prefix:
  - anthropic/*  → Anthropic Messages API (with format conversion)
  - deepseek/*   → DeepSeek API (OpenAI-compatible)

Usage:
    # Put your keys in files:
    echo "sk-ant-..." > .anthropickey
    echo "sk-..."     > .deepseekkey

    # Start proxy:
    python capx/serving/multi_provider_server.py --port 8110

    # Run evaluation with Claude:
    MUJOCO_GL=glfw python capx/envs/launch.py \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \\
        --total-trials 1 --num-workers 1 \\
        --model "anthropic/claude-sonnet-4" \\
        --server-url "http://127.0.0.1:8110/chat/completions"

    # Or with DeepSeek:
    MUJOCO_GL=glfw python capx/envs/launch.py \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \\
        --total-trials 1 --num-workers 1 \\
        --model "deepseek/deepseek-chat" \\
        --server-url "http://127.0.0.1:8110/chat/completions"
"""

import logging
import time
from pathlib import Path
from typing import Literal

import httpx
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Pydantic models (OpenAI chat completions format) ──────────────────────


class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentItem] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = 0.2
    max_tokens: int | None = 256
    stream: bool = False
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None
    thinking: dict | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_key(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Key file not found: {path}")
    for line in p.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise ValueError(f"No key found in {path}")


# ── Anthropic adapter ─────────────────────────────────────────────────────


def _convert_messages_to_anthropic(
    messages: list[Message],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, messages_list).
    """
    system = None
    out: list[dict] = []

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system = msg.content
            elif isinstance(msg.content, list):
                system = " ".join(
                    item.text for item in msg.content if item.type == "text" and item.text
                )
            continue

        if isinstance(msg.content, str):
            content: str | list[dict] = msg.content
        elif isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text" and item.text:
                    content.append({"type": "text", "text": item.text})
                elif item.type == "image_url" and item.image_url:
                    url = item.image_url.url
                    if url.startswith("data:"):
                        meta, data = url.split(",", 1)
                        media_type = meta.split(":")[1].split(";")[0]
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                    else:
                        content.append(
                            {"type": "image", "source": {"type": "url", "url": url}}
                        )
        else:
            content = msg.content or ""

        out.append({"role": msg.role, "content": content})

    return system, out


async def _call_anthropic(
    client: httpx.AsyncClient,
    api_key: str,
    request: ChatCompletionRequest,
    base_url: str = "https://api.anthropic.com",
) -> ChatCompletionResponse:
    model = request.model
    if model.startswith("anthropic/"):
        model = model[len("anthropic/"):]

    system, messages = _convert_messages_to_anthropic(request.messages)

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": request.max_tokens or 4096,
    }
    if system:
        body["system"] = system
    if request.thinking:
        body["thinking"] = request.thinking
    elif request.temperature is not None:
        body["temperature"] = request.temperature

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    url = f"{base_url.rstrip('/')}/v1/messages"
    logger.info(f"Anthropic request: model={model}, messages={len(messages)}, max_tokens={body.get('max_tokens')}")

    import asyncio as _aio
    last_error = ""
    for attempt in range(3):
        resp = await client.post(
            url,
            headers=headers,
            json=body,
            timeout=300.0,
        )
        if resp.status_code == 200:
            break
        last_error = resp.text
        logger.warning(f"Anthropic API attempt {attempt+1} failed ({resp.status_code}): {last_error}")
        if resp.status_code < 500 and resp.status_code != 429:
            raise HTTPException(status_code=resp.status_code, detail=f"Anthropic API error: {last_error}")
        await _aio.sleep(2 ** attempt)
    else:
        raise HTTPException(status_code=resp.status_code, detail=f"Anthropic API error after retries: {last_error}")

    data = resp.json()

    text_parts = [block["text"] for block in data.get("content", []) if block.get("type") == "text"]
    content = "\n".join(text_parts)

    return ChatCompletionResponse(
        id=data.get("id", "msg_unknown"),
        created=int(time.time()),
        model=data.get("model", model),
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


# ── App factory ────────────────────────────────────────────────────────────


def create_app(
    anthropic_key: str | None,
    deepseek_key: str | None,
    anthropic_base_url: str = "https://api.anthropic.com",
) -> FastAPI:
    app = FastAPI(title="Multi-Provider LLM Proxy", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    anthropic_http = httpx.AsyncClient() if anthropic_key else None

    deepseek_client = (
        AsyncOpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
        if deepseek_key
        else None
    )

    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        model = request.model
        logger.info(f"Request: model={model}, msgs={len(request.messages)}, "
                     f"max_tokens={request.max_tokens}, stream={request.stream}")

        # ── Anthropic ──
        if model.startswith("anthropic/"):
            if not anthropic_key or not anthropic_http:
                raise HTTPException(400, "Anthropic API key not configured")
            return await _call_anthropic(anthropic_http, anthropic_key, request, anthropic_base_url)

        # ── DeepSeek ──
        if model.startswith("deepseek/"):
            if not deepseek_client:
                raise HTTPException(400, "DeepSeek API key not configured")

            ds_model = model[len("deepseek/"):]
            kwargs: dict = {
                "model": ds_model,
                "messages": [m.model_dump(exclude_none=True) for m in request.messages],
                "stream": False,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens

            try:
                response = await deepseek_client.chat.completions.create(**kwargs)
            except Exception as e:
                raise HTTPException(500, f"DeepSeek API error: {e}")

            return ChatCompletionResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=c.index,
                        message=Message(role=c.message.role, content=c.message.content),
                        finish_reason=c.finish_reason,
                    )
                    for c in response.choices
                ],
            )

        raise HTTPException(
            400,
            f"Unknown provider for model '{model}'. Use anthropic/ or deepseek/ prefix.",
        )

    @app.get("/health")
    async def health():
        providers = []
        if anthropic_key:
            providers.append("anthropic")
        if deepseek_key:
            providers.append("deepseek")
        return {"status": "ok", "providers": providers}

    return app


# ── CLI ────────────────────────────────────────────────────────────────────


def main(
    anthropic_key_file: str = ".anthropickey",
    deepseek_key_file: str = ".deepseekkey",
    anthropic_key: str | None = None,
    deepseek_key: str | None = None,
    anthropic_base_url: str = "https://mapi.darkjason.com",
    host: str = "0.0.0.0",
    port: int = 8110,
):
    """Multi-provider LLM proxy. Routes by model prefix (anthropic/, deepseek/)."""

    if anthropic_key is None:
        try:
            anthropic_key = _load_key(anthropic_key_file)
            print(f"Loaded Anthropic key from {anthropic_key_file}")
        except FileNotFoundError:
            print(f"Anthropic key file not found: {anthropic_key_file} (skipped)")

    if deepseek_key is None:
        try:
            deepseek_key = _load_key(deepseek_key_file)
            print(f"Loaded DeepSeek key from {deepseek_key_file}")
        except FileNotFoundError:
            print(f"DeepSeek key file not found: {deepseek_key_file} (skipped)")

    if not anthropic_key and not deepseek_key:
        raise ValueError(
            "At least one API key required. Create .anthropickey and/or .deepseekkey"
        )

    active = []
    if anthropic_key:
        active.append("anthropic")
    if deepseek_key:
        active.append("deepseek")
    print(f"Active providers: {', '.join(active)}")
    print(f"Listening on http://{host}:{port}/chat/completions")

    app = create_app(anthropic_key=anthropic_key, deepseek_key=deepseek_key, anthropic_base_url=anthropic_base_url)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
