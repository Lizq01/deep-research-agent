"""独立评审器（LLM-as-Judge）：评测流水线产出质量。

与流水线内的 Reviewer 职责分离（避免"自己写的自己评"偏差）：
- Reviewer 在环内，反馈给 Writer 用于修订，rubric 面向可执行修改意见；
- Judge 在环外，只打分不改稿，维度面向评测结论（覆盖度 / 忠实度 / 结构 / 洞察），
  且拿着评测集预置的"应覆盖要点清单"逐项核对——这是 Reviewer 没有的信息。
"""
from src.llm import chat
from src.utils import extract_json_object

_SYSTEM = "你是独立的质量评审员，只输出 JSON，不输出任何其他文字。"

# 每条证据给评审员看的正文字符数（控制 prompt 体积，标题全量保留）
_EV_CHARS = 600

_PROMPT = """请评审下面这份调研简报。

调研主题：{topic}
出题时设定的"应覆盖要点"清单（覆盖度逐项核对用）：
{aspects}

评审可参考的证据列表（共 {n_evidence} 条，正文已截断）：
{evidence}

评分维度（每项 0~10 分）：
1. coverage 覆盖度：要点清单逐项核对，覆盖越全分越高；同时关注是否跑题
2. grounding 忠实度：关键论断是否有证据支撑，是否存在编造（幻觉）
3. structure 结构：一句话结论、摘要、分析小节、来源引用是否齐备规范
4. insight 洞察：是否有综合分析与增量结论，而非资料堆砌

只输出如下 JSON，不要输出任何其他文字：
{{"scores": {{"coverage": 0, "grounding": 0, "structure": 0, "insight": 0}}, "overall": 0到10, "missing_aspects": ["未覆盖的要点"], "hallucination_risk": "low或medium或high", "comment": "一句话总评"}}

overall 是你的整体判断，不必等于四项平均。

调研简报正文：
{report}"""


def _format_aspects(aspects: list[str]) -> str:
    if not aspects:
        return "（本题未设定要点清单，coverage 按主题相关性评）"
    return "\n".join(f"- {a}" for a in aspects)


def _format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "（无证据——简报应如实说明检索失败，此时 grounding 按'未编造'评）"
    rows = []
    for i, e in enumerate(evidence, start=1):
        content = (e.get("content") or "")[:_EV_CHARS]
        rows.append(f"[{i}] {e.get('title', '')}\n{content}")
    return "\n\n".join(rows)


def judge_report(
    topic: str, aspects: list[str], evidence: list[dict], report: str
) -> dict:
    """返回 {"scores": {...}, "overall": int, "missing_aspects": [...], ...}

    解析失败时返回 parse_error=True 且分数为 None——评测聚合时跳过该维度，
    不默认给分（避免把"评审员没输出 JSON"混成"报告质量差"）。
    """
    raw = chat(
        _PROMPT.format(
            topic=topic,
            aspects=_format_aspects(aspects),
            n_evidence=len(evidence),
            evidence=_format_evidence(evidence),
            report=report,
        ),
        system=_SYSTEM,
        temperature=0.1,
    )

    data = extract_json_object(raw)
    if not data:
        return {
            "scores": None,
            "overall": None,
            "missing_aspects": [],
            "hallucination_risk": "unknown",
            "comment": "",
            "parse_error": True,
        }

    scores_raw = data.get("scores") or {}
    scores = {}
    for k in ("coverage", "grounding", "structure", "insight"):
        v = scores_raw.get(k)
        scores[k] = int(v) if isinstance(v, (int, float)) else None

    overall = data.get("overall")
    return {
        "scores": scores,
        "overall": int(overall) if isinstance(overall, (int, float)) else None,
        "missing_aspects": [str(a) for a in data.get("missing_aspects", [])],
        "hallucination_risk": str(data.get("hallucination_risk", "unknown")),
        "comment": str(data.get("comment", "")).strip(),
        "parse_error": False,
    }
