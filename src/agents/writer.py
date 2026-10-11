"""Writer：把证据整合成一份带引用编号的 Markdown 简报。"""
from src.llm import chat

_SYSTEM = "你是一位资深行业研究员，善于基于给定资料写出结构清晰、引用严谨的调研简报。"

_PROMPT = """请基于下面的检索资料，围绕调研主题写一份 Markdown 格式的调研简报。

主题：{topic}

要求：
1. 结构：一句话结论 → 摘要 → 3~5 个分析小节 → 关键发现
2. 每个关键论断后面用 [编号] 标注来源，编号对应下方资料（如 [1]、[2]）
3. 只使用资料中的信息，资料没提到的不要编造；确实没覆盖的点放到"局限与待补充"
4. 全文使用中文

检索资料：
{evidence}"""


def write_report(topic: str, evidence: list[dict]) -> str:
    if evidence:
        evidence_text = "\n\n".join(
            f"[{i}] {e['title']}\nURL: {e['url']}\n内容: {e['content']}"
            for i, e in enumerate(evidence, start=1)
        )
    else:
        evidence_text = "（本次没有检索到任何资料，请如实说明情况，不要编造内容。）"

    return chat(
        _PROMPT.format(topic=topic, evidence=evidence_text),
        system=_SYSTEM,
        temperature=0.4,
    )
