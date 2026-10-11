"""LLM 客户端封装：任何 OpenAI 兼容接口都能用（DeepSeek / OpenAI / Kimi / Qwen ...）。"""
from openai import OpenAI

from src.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL

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
    """一次最简单的对话调用。M2 里会替换为带工具调用的 Agent 节点。"""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()
