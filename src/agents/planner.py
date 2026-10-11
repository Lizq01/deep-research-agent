"""Planner：把调研主题拆解成若干检索子问题。"""
from src.llm import chat
from src.utils import extract_json_list

_SYSTEM = "你是一个严谨的调研规划助手，擅长把一个调研主题拆解成若干互补的检索子问题。"

_PROMPT = """请把下面的调研主题拆解成 {n} 个高质量检索关键词/子问题。

要求：
1. 覆盖不同角度（现状、关键技术、代表玩家、争议与风险等）
2. 每个子问题都要适合直接丢进搜索引擎
3. 只输出 JSON 字符串数组，例如：["关键词1", "关键词2"]
4. 不要输出任何其他文字

调研主题：{topic}"""


def make_plan(topic: str, n: int = 4) -> list[str]:
    """返回 n 个检索子问题；解析失败时退化为原始主题。"""
    raw = chat(_PROMPT.format(n=n, topic=topic), system=_SYSTEM)
    queries = extract_json_list(raw)
    return queries[:n] if queries else [topic]
