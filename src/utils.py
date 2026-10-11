"""小工具函数集。"""
import json
import re


def extract_json_list(text: str) -> list[str]:
    """从 LLM 输出中提取 JSON 字符串数组。

    兼容三种情况：纯 JSON、```json 代码块、前后带解释文字。
    """
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()

    match = re.search(r"\[.*\]", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except json.JSONDecodeError:
            pass

    # 兜底：按行拆分，去掉序号和符号
    lines = [ln.strip(" -\t") for ln in text.splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("[")]
