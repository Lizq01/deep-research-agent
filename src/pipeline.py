"""M1 主流程：规划 → 检索 → 撰写（顺序执行）。

说明：这三个步骤会和 M2 的 LangGraph 节点一一对应，
迁移成 StateGraph 时只需要把函数包成节点即可。
"""
from datetime import datetime

from src import tracing
from src.agents.planner import make_plan
from src.agents.researcher import gather_evidence
from src.agents.writer import write_report
from src.config import MAX_QUERIES, OUTPUT_DIR


def run(topic: str, log=print) -> str:
    """执行一次完整调研，返回生成的报告文件路径。"""
    log(f"[1/3] Planner 拆解主题：{topic}")
    with tracing.span("planner") as s:
        queries = make_plan(topic, n=MAX_QUERIES)
        s.set("queries", queries)
    for i, q in enumerate(queries, 1):
        log(f"      {i}. {q}")

    log("[2/3] Researcher 检索并抓取资料 ...")
    evidence = gather_evidence(queries)
    log(f"      共收集 {len(evidence)} 条资料")
    if t := tracing.current():
        t.add_event("node", "research", {"hits": len(evidence)})

    log("[3/3] Writer 生成简报 ...")
    with tracing.span("writer", {"evidence": len(evidence)}):
        report = write_report(topic, evidence)

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"report_{stamp}.md"

    sources = "\n".join(
        f"{i}. [{e['title']}]({e['url']})" for i, e in enumerate(evidence, start=1)
    )
    content = f"# 调研简报：{topic}\n\n{report}\n\n---\n\n## 参考来源\n\n{sources}\n"
    path.write_text(content, encoding="utf-8")

    log(f"      报告已保存：{path}")
    return str(path)
