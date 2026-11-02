"""core · LLM 客户端接口。

统一契约：一次带可选系统提示的对话调用，返回纯文本。
services 层只认这个接口，不感知底层是 DeepSeek / OpenAI / Kimi / Qwen 还是本地模型。
"""
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """LLM 客户端抽象：屏蔽厂商差异，只暴露 chat 能力。"""

    @abstractmethod
    def chat(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """发起一次对话，返回模型输出的纯文本。

        实现侧需保证两点：
        - 带显式超时 + 有限重试（指数退避 + 抖动），重试耗尽后抛出带上下文的异常；
        - 每次调用（含失败）上报 trace，便于归因 token 用量与延迟。
        """
