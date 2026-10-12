"""LLM 客户端封装：任何 OpenAI 兼容接口都能用（DeepSeek / OpenAI / Kimi / Qwen ...）。

所有调用带显式超时 + 有限重试（指数退避 + 抖动），重试耗尽后抛出带上下文的异常。
"""
import random
import time

from openai import OpenAI

from src.config import (
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "未找到 OPENAI_API_KEY。\n"
                "请把 .env.example 复制为 .env，并填入你的 API Key（详见 README.md）。"
            )
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def chat(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    """一次最简单的对话调用（带超时与重试，重试耗尽后抛 RuntimeError）。"""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                timeout=LLM_TIMEOUT,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # 网络抖动 / 限流 / 超时均可重试
            last_err = e
            if attempt < LLM_MAX_RETRIES:
                # 指数退避 + 随机抖动，避免多节点同时重试打爆接口
                time.sleep(min(2**attempt + random.random(), 5.0))

    raise RuntimeError(
        f"LLM 调用失败（已重试 {LLM_MAX_RETRIES} 次）：{last_err}"
    ) from last_err
