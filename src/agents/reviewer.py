"""Reviewer：对简报做结构化质量评审，输出 JSON 结论（passed/score/issues/feedback）。

设计要点（面试可讲）：
- 评审标准是 5 维 rubric，避免"感觉不够好"式的模糊反馈——
  Evaluator-Optimizer 模式要求评估者给出**具体可执行**的修改意见；
- passed 有硬判定线（score >= 8 且无幻觉问题），不交给模型自由发挥；
- JSON 解析失败时按"通过 + parse_error 标记"兜底：回退修订本身有
  MAX_REVISIONS 上限，兜底策略不会导致死循环。
"""
from src.llm import chat
from src.utils import extract_json_object

_SYSTEM = "你是一位严苛的主编，负责评审调研简报的质量，只输出 JSON，不输出任何其他文字。"

_PROMPT = """请评审下面这份调研简报的质量。

调研主题：{topic}
引用资料条数：{n_evidence}

评审 rubric（每项 0~2 分，总分 10）：
1. 结构完整：是否有一句话结论、摘要、分析小节、关键发现
2. 引用规范：关键论断是否都有 [编号] 标注，编号是否都在 1~{n_evidence} 范围内
3. 忠于资料：是否存在资料中找不到依据的论断（幻觉）
4. 信息增量：是否有实质分析，而非简单堆砌资料
5. 语言质量：中文表达是否通顺专业

只输出如下 JSON，不要输出任何其他文字：
{{"passed": true或false, "score": 0到10的整数, "issues": ["问题1", "问题2"], "feedback": "给作者的具体可执行修改意见，逐条列出"}}

passed 判定标准：score >= 8 且没有"幻觉"类问题。

调研简报：
{report}"""


def review_report(topic: str, evidence: list[dict], report: str) -> dict:
    """返回 {"passed": bool, "score": int, "issues": [...], "feedback": str, ...}"""
    raw = chat(
        _PROMPT.format(topic=topic, n_evidence=len(evidence), report=report),
        system=_SYSTEM,
        temperature=0.1,
    )

    data = extract_json_object(raw)
    if not data:
        # 解析失败兜底：不阻塞流程（修订次数有上限），但打上标记便于观测
        return {
            "passed": True,
            "score": -1,
            "issues": [],
            "feedback": "",
            "parse_error": True,
        }

    return {
        "passed": bool(data.get("passed", False)),
        "score": int(data.get("score", 0) or 0),
        "issues": [str(i) for i in data.get("issues", [])],
        "feedback": str(data.get("feedback", "")).strip(),
        "parse_error": False,
    }
