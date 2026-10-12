"""M2 主流程：LangGraph 多智能体编排。

图结构（面试叙事：Chaining + Parallelization + Evaluator-Optimizer + 人工介入）：

    planner ──Send ×N──> researcher ×N（每个子问题一个，并行）──> aggregator
                                                                    │
                                              ┌── 带反馈修订 <──── writer
                                              │                     │
                                              │                   reviewer（结构化评审）
                                              │                     │
                                              │     不通过且修订数 ≤ MAX_REVISIONS
                                              └─────────────────────┤
                                                                    │ 通过 / 达上限
                                                                human_review（interrupt 挂起）
                                                                    │ y 通过 / 修改意见 / n 放弃
                                                                save_report ──> END

有界自主（C5）：
- Writer↔Reviewer 回退修订上限 MAX_REVISIONS（默认 2），超限强制转人工；
- 图执行设 recursion_limit=50，任何异常循环都会被 LangGraph 强制终止；
- 人工介入点是唯一的发布闸门，报告未经确认不会落盘。
"""
import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from rich.console import Console

from src.agents.planner import make_plan
from src.agents.reviewer import review_report
from src.agents.writer import write_report
from src.config import (
    MAX_PAGES,
    MAX_QUERIES,
    MAX_REVISIONS,
    OUTPUT_DIR,
    PAGES_PER_QUERY,
)
from src.tools import fetch_page, web_search

_console = Console()


def _log(msg: str) -> None:
    _console.print(msg)


# ===== 状态定义 =====


class Evidence(TypedDict):
    title: str
    url: str
    content: str
    query: str  # 记录来源子问题，便于追溯


class ResearchState(TypedDict, total=False):
    topic: str
    queries: list[str]
    # 并行 Researcher 用 operator.add 归并各自的检索结果
    raw_evidence: Annotated[list[Evidence], operator.add]
    evidence: list[Evidence]  # aggregator 去重截断后的最终证据
    report: str
    feedback: str  # Reviewer / 人工给 Writer 的修改意见
    review: dict
    revision_count: int
    aborted: bool
    final_path: str


# ===== 节点 =====


def planner_node(state: ResearchState) -> dict:
    _log(f"[1/5] Planner 拆解主题：{state['topic']}")
    queries = make_plan(state["topic"], n=MAX_QUERIES)
    for i, q in enumerate(queries, 1):
        _log(f"      {i}. {q}")
    return {"queries": queries}


def fan_out_to_researchers(state: ResearchState) -> list[Send]:
    """动态并行：每个子问题派发一个独立 Researcher。"""
    _log(f"[2/5] Researcher ×{len(state['queries'])} 并行检索 ...")
    return [Send("researcher", {"query": q}) for q in state["queries"]]


def researcher_node(payload: dict) -> dict:
    """单个子问题的完整检索链路：搜索 → 抓正文。失败不抛异常，返回空结果。"""
    query = payload["query"]
    items: list[Evidence] = []
    for r in web_search(query, max_results=4)[:PAGES_PER_QUERY]:
        url = r["url"]
        if not url:
            continue
        content = fetch_page(url) or r["snippet"]  # 抓不到正文退回摘要兜底
        items.append(
            {"title": r["title"], "url": url, "content": content, "query": query}
        )
    _log(f"      · {query} → {len(items)} 条")
    return {"raw_evidence": items}


def aggregator_node(state: ResearchState) -> dict:
    """跨分支归并：按 URL 去重 + 按 MAX_PAGES 截断。"""
    seen: set[str] = set()
    evidence: list[Evidence] = []
    for e in state.get("raw_evidence", []):
        if e["url"] not in seen:
            seen.add(e["url"])
            evidence.append(e)
    evidence = evidence[:MAX_PAGES]
    _log(f"[3/5] Aggregator 归并：去重后 {len(evidence)} 条资料")
    return {"evidence": evidence}


def writer_node(state: ResearchState) -> dict:
    revision = state.get("revision_count", 0)
    feedback = state.get("feedback", "")
    tag = "撰写初稿" if revision == 0 else f"第 {revision} 次修订"
    _log(f"[4/5] Writer {tag} ...")
    report = write_report(state["topic"], state.get("evidence", []), feedback=feedback)
    return {"report": report, "revision_count": revision + 1}


def reviewer_node(state: ResearchState) -> dict:
    _log("[5/5] Reviewer 评审 ...")
    review = review_report(
        state["topic"], state.get("evidence", []), state["report"]
    )
    if review.get("parse_error"):
        _log("      [yellow]评审输出解析失败，按通过处理（已标记）[/yellow]")
    else:
        status = "[green]通过[/green]" if review["passed"] else "[red]不通过[/red]"
        _log(f"      得分 {review['score']}/10 → {status}")
        for issue in review["issues"]:
            _log(f"      - {issue}")
    return {"review": review, "feedback": review.get("feedback", "")}


def route_after_review(state: ResearchState) -> str:
    review = state.get("review") or {}
    if review.get("passed"):
        return "human_review"
    if state.get("revision_count", 0) > MAX_REVISIONS:
        _log(
            f"      [yellow]已达修订上限（{MAX_REVISIONS} 次），强制转人工审核[/yellow]"
        )
        return "human_review"
    _log("      ↩ 带评审意见退回 Writer 修订")
    return "writer"


def human_review_node(
    state: ResearchState,
) -> Command[Literal["save_report", "writer", "__end__"]]:
    """interrupt 挂起：把报告与评审结论交给人决策，恢复后按决策路由。"""
    decision = interrupt(
        {
            "type": "report_review",
            "topic": state["topic"],
            "review": state.get("review", {}),
            "revision_count": state.get("revision_count", 0),
            "report": state["report"],
        }
    )
    action = str(decision.get("action", "")).strip().lower()
    if action == "approve":
        return Command(goto="save_report")
    if action == "revise":
        comment = str(decision.get("comment", "")).strip()
        _log(f"      ↩ 人工退回修订：{comment}")
        return Command(
            goto="writer", update={"feedback": f"【人工修改意见】{comment}"}
        )
    return Command(goto=END, update={"aborted": True})


def save_report_node(state: ResearchState) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"report_{stamp}.md"

    evidence = state.get("evidence", [])
    sources = "\n".join(
        f"{i}. [{e['title']}]({e['url']})" for i, e in enumerate(evidence, start=1)
    )
    review = state.get("review", {})
    meta = (
        f"<!-- M2 | 评审得分: {review.get('score', 'N/A')}/10"
        f" | 修订次数: {state.get('revision_count', 0) - 1} -->\n\n"
    )
    content = (
        f"# 调研简报：{state['topic']}\n\n{meta}{state['report']}"
        f"\n\n---\n\n## 参考来源\n\n{sources}\n"
    )
    path.write_text(content, encoding="utf-8")
    _log(f"      报告已保存：{path}")
    return {"final_path": str(path)}


# ===== 图构建 =====


def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("aggregator", aggregator_node)
    g.add_node("writer", writer_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("human_review", human_review_node)
    g.add_node("save_report", save_report_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out_to_researchers, ["researcher"])
    g.add_edge("researcher", "aggregator")
    g.add_edge("aggregator", "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review, ["writer", "human_review"])
    g.add_edge("save_report", END)

    # 检查点用于 interrupt 挂起/恢复；CLI 单次运行用内存版即可
    return g.compile(checkpointer=InMemorySaver())
