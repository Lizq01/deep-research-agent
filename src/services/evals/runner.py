"""评测 runner：批量运行调研流水线 → 确定性指标 + LLM 独立评审 → 聚合报告。

指标分两层，刻意区分：
- 确定性指标（无 LLM，可复现）：引用编号有效性（正则校验）、证据数、报告长度、
  RAG 上下文压缩率、耗时、token 用量、重试次数；
- 评审指标（LLM-as-Judge，有随机性）：覆盖度 / 忠实度 / 结构 / 洞察 + 总分。

产物：
- outputs/eval_results_<stamp>.json ：逐题原始数据（含 judge 明细、trace 文件路径）
- outputs/eval_report_<stamp>.md   ：人读的聚合报告
"""
import json
import re
from datetime import datetime
from pathlib import Path

from src import tracing
from src.config import EVAL_DATASET, OUTPUT_DIR
from src.evals.judge import judge_report

_CITE_RE = re.compile(r"\[(\d+)\]")


# ===== 数据集 =====


def load_dataset(path: Path = EVAL_DATASET) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    topics = doc.get("topics", [])
    if not topics:
        raise ValueError(f"评测集为空或格式不对：{path}")
    return doc


def pick_topics(n: int | None = None, all_: bool = False) -> list[dict]:
    """选取评测样本：all_ 全量；否则取前 n 条（数据集顺序即默认抽样，可复现）。"""
    topics = load_dataset()["topics"]
    if all_ or n is None or n >= len(topics):
        return topics
    return topics[:n]


# ===== 确定性指标 =====


def citation_metrics(report: str, n_evidence: int) -> dict:
    """引用编号有效性：[n] 必须落在 1~n_evidence 内（不耗 LLM，可复现）。"""
    cites = [int(m) for m in _CITE_RE.findall(report)]
    valid = [c for c in cites if 1 <= c <= n_evidence] if n_evidence else []
    return {
        "citations_total": len(cites),
        "citations_valid": len(valid),
        "citations_invalid": len(cites) - len(valid),
        "cited_sources": len(set(valid)),
        "source_coverage": round(len(set(valid)) / n_evidence, 2) if n_evidence else 0.0,
    }


def parse_report_file(path: str | Path) -> tuple[str, int]:
    """从落盘报告解析（正文, 证据条数）。M1 对照走这条路。"""
    text = Path(path).read_text(encoding="utf-8")
    body = text
    n = 0
    if "## 参考来源" in text:
        body, _, tail = text.partition("## 参考来源")
        n = len(re.findall(r"^\d+\. \[", tail, flags=re.M))
    return body.rstrip("-\n "), n


# ===== 流水线执行 =====


def _run_m2_once(topic: str, log=print) -> dict:
    """跑一次 M2 图（评测模式：人工闸门自动通过），返回最终 state。"""
    from langgraph.types import Command

    from src.graph import build_graph

    graph = build_graph()
    thread_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    result = graph.invoke({"topic": topic}, config=config)
    while interrupts := result.get("__interrupt__"):
        log("      · 评测模式：人工闸门自动通过")
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    return result


def eval_one_m2(item: dict, log=print) -> dict:
    """对单条主题跑 M2 并完成全部指标采集。"""
    topic = item["topic"]
    log(f"▶ [{item['id']}] {topic}")

    tracing.start_run(topic, "eval")
    try:
        state = _run_m2_once(topic, log=log)
    except Exception as e:
        summary = tracing.finish("error", {"error": str(e)})
        log(f"      ✗ 运行失败：{e}")
        return {"id": item["id"], "topic": topic, "status": "error",
                "error": str(e), "trace": summary.get("_trace_file")}
    status = "aborted" if state.get("aborted") else "ok"
    summary = tracing.finish(status, {"final_path": state.get("final_path")})

    evidence = state.get("rag_evidence") or state.get("evidence", [])
    report = state.get("report", "")
    review = state.get("review", {})
    rag = state.get("rag_stats", {}) or {}

    judge = judge_report(topic, item.get("expected_aspects", []), evidence, report)

    return {
        "id": item["id"],
        "topic": topic,
        "status": status,
        "report_path": state.get("final_path"),
        "report_chars": len(report),
        "n_evidence": len(evidence),
        "review_score": review.get("score"),
        "revisions": max(state.get("revision_count", 1) - 1, 0),
        "rag": {
            "mode": rag.get("mode"),
            "chunks": rag.get("chunks"),
            "hits": rag.get("hits"),
            "input_chars": rag.get("input_chars"),
            "output_chars": rag.get("output_chars"),
            "compression": (
                round((rag["input_chars"] - rag["output_chars"]) / rag["input_chars"], 2)
                if rag.get("input_chars")
                else None
            ),
        },
        "runtime": {
            "duration_s": summary.get("duration_s"),
            "llm_calls": summary["llm"]["calls"],
            "total_tokens": summary["llm"]["total_tokens"],
            "llm_retries": summary["llm"]["retries"],
        },
        **citation_metrics(report, len(evidence)),
        "judge": judge,
        "trace": summary.get("_trace_file"),
    }


def eval_one_m1(item: dict, log=print) -> dict:
    """对单条主题跑 M1 顺序流程（对照用，无 RAG / 无评审环）。"""
    from src.pipeline import run as run_m1_pipeline

    topic = item["topic"]
    log(f"▶ [M1·{item['id']}] {topic}")
    tracing.start_run(topic, "eval-m1")
    try:
        path = run_m1_pipeline(topic, log=log)
    except Exception as e:
        summary = tracing.finish("error", {"error": str(e)})
        return {"id": item["id"], "topic": topic, "status": "error", "error": str(e),
                "trace": summary.get("_trace_file")}
    summary = tracing.finish("ok", {"final_path": path})

    body, n_ev = parse_report_file(path)
    judge = judge_report(topic, item.get("expected_aspects", []), [], body)
    # M1 的证据在落盘报告里只有来源清单，judge 拿不到证据正文 → grounding 仅供参考
    return {
        "id": item["id"],
        "topic": topic,
        "status": "ok",
        "report_path": str(path),
        "report_chars": len(body),
        "n_evidence": n_ev,
        "runtime": {
            "duration_s": summary.get("duration_s"),
            "llm_calls": summary["llm"]["calls"],
            "total_tokens": summary["llm"]["total_tokens"],
            "llm_retries": summary["llm"]["retries"],
        },
        **citation_metrics(body, n_ev),
        "judge": judge,
        "trace": summary.get("_trace_file"),
    }


# ===== 聚合与报告 =====


def _mean(values: list, ndigits: int = 2):
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), ndigits) if vals else None


def _agg_dim(results: list[dict], key: str):
    return _mean(
        [(r.get("judge", {}).get("scores") or {}).get(key) for r in results]
    )


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "judge_overall": _mean([r["judge"].get("overall") for r in ok]),
        "dim_coverage": _agg_dim(ok, "coverage"),
        "dim_grounding": _agg_dim(ok, "grounding"),
        "dim_structure": _agg_dim(ok, "structure"),
        "dim_insight": _agg_dim(ok, "insight"),
        "citation_validity": _mean(
            [
                r["citations_valid"] / r["citations_total"]
                for r in ok
                if r.get("citations_total")
            ]
        ),
        "source_coverage": _mean([r.get("source_coverage") for r in ok]),
        "mean_evidence": _mean([r.get("n_evidence") for r in ok]),
        "mean_report_chars": _mean([r.get("report_chars") for r in ok]),
        "mean_revisions": _mean([r.get("revisions", 0) for r in ok]),
        "mean_duration_s": _mean([r["runtime"]["duration_s"] for r in ok]),
        "mean_tokens": _mean([r["runtime"]["total_tokens"] for r in ok], 0),
        "llm_retries_total": sum(
            r["runtime"].get("llm_retries", 0) for r in ok
        ),
        "rag_compression": _mean([r.get("rag", {}).get("compression") for r in ok]),
        "parse_errors": sum(1 for r in ok if r["judge"].get("parse_error")),
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".") if v % 1 else str(int(v))
    return str(v)


def write_report(
    results: list[dict], agg: dict, m1_results: list[dict] | None = None
) -> tuple[str, str]:
    """写 eval_report_*.md 与 eval_results_*.json，返回两个路径。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = OUTPUT_DIR / f"eval_report_{stamp}.md"
    json_path = OUTPUT_DIR / f"eval_results_{stamp}.json"
    OUTPUT_DIR.mkdir(exist_ok=True)

    lines = [
        "# 评测报告",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本：{agg['n_ok']}/{agg['n_total']} 条主题完成"
        f"（评测失败 {agg['n_total'] - agg['n_ok']} 条）",
        f"- 评审器说明：LLM-as-Judge 独立评审（与流水线内 Reviewer 分离），"
        f"解析失败 {agg['parse_errors']} 条",
        "",
        "## 总体指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| Judge 总分 | {_fmt(agg['judge_overall'])}/10 | 四维之外的整体判断 |",
        f"| 覆盖度 coverage | {_fmt(agg['dim_coverage'])}/10 | 对照要点清单逐项核对 |",
        f"| 忠实度 grounding | {_fmt(agg['dim_grounding'])}/10 | 幻觉风险反向指标 |",
        f"| 结构 structure | {_fmt(agg['dim_structure'])}/10 | |",
        f"| 洞察 insight | {_fmt(agg['dim_insight'])}/10 | |",
        f"| 引用有效率 | {_fmt(agg['citation_validity'])} | [编号] 落在证据范围内比例（确定性指标） |",
        f"| 证据引用覆盖 | {_fmt(agg['source_coverage'])} | 被引用过的证据占比 |",
        f"| 平均证据数 | {_fmt(agg['mean_evidence'])} 条 | |",
        f"| 平均报告长度 | {_fmt(agg['mean_report_chars'])} 字符 | |",
        f"| 平均修订次数 | {_fmt(agg['mean_revisions'])} | Writer↔Reviewer 回退 |",
        f"| RAG 上下文压缩 | {_fmt(agg['rag_compression'])} | 输入字符缩减比例 |",
        f"| 平均耗时 | {_fmt(agg['mean_duration_s'])} s | 单主题端到端 |",
        f"| 平均 token | {_fmt(agg['mean_tokens'])} | 输入+输出 |",
        f"| LLM 重试总数 | {_fmt(agg['llm_retries_total'])} 次 | |",
        "",
        "## 分主题明细",
        "",
        "| # | 主题 | Judge | 覆盖 | 忠实 | 结构 | 洞察 | 引用有效率 | 证据 | RAG压缩 | 耗时s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        j = r.get("judge", {})
        s = j.get("scores") or {}
        rag = r.get("rag", {})
        if r.get("status") != "ok":
            lines.append(
                f"| {r['id']} | {r['topic'][:24]} | 失败 | - | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {r['id']} | {r['topic'][:24]} | {_fmt(j.get('overall'))} | "
            f"{_fmt(s.get('coverage'))} | {_fmt(s.get('grounding'))} | "
            f"{_fmt(s.get('structure'))} | {_fmt(s.get('insight'))} | "
            f"{_fmt(r['citations_valid'] / r['citations_total']) if r['citations_total'] else '-'} | "
            f"{r['n_evidence']} | {_fmt(rag.get('compression'))} | "
            f"{_fmt(r['runtime']['duration_s'])} |"
        )

    lines += ["", "## 评审摘录", ""]
    for r in results:
        j = r.get("judge", {})
        if j.get("comment"):
            missing = (
                f"（未覆盖：{'；'.join(j['missing_aspects'][:3])}）"
                if j.get("missing_aspects")
                else ""
            )
            lines.append(f"- **{r['id']}** {j['comment']} {missing}")
        risk = j.get("hallucination_risk")
        if risk and risk not in {"low", "unknown"}:
            lines.append(f"  - ⚠ 幻觉风险评级：{risk}")

    if m1_results:
        m1_agg = aggregate(m1_results)
        lines += [
            "",
            "## M1 对照（同题不同流程）",
            "",
            "| 指标 | M1 顺序流程 | M2 多智能体+RAG |",
            "|---|---|---|",
            f"| Judge 总分 | {_fmt(m1_agg['judge_overall'])} | {_fmt(agg['judge_overall'])} |",
            f"| 覆盖度 | {_fmt(m1_agg['dim_coverage'])} | {_fmt(agg['dim_coverage'])} |",
            f"| 引用有效率 | {_fmt(m1_agg['citation_validity'])} | {_fmt(agg['citation_validity'])} |",
            f"| 平均证据数 | {_fmt(m1_agg['mean_evidence'])} | {_fmt(agg['mean_evidence'])} |",
            f"| 平均耗时 s | {_fmt(m1_agg['mean_duration_s'])} | {_fmt(agg['mean_duration_s'])} |",
            f"| 平均 token | {_fmt(m1_agg['mean_tokens'])} | {_fmt(agg['mean_tokens'])} |",
            "",
            "> 注意：M1 的 Judge 评审拿不到证据正文（只有报告本身），grounding 维度",
            "> 不可直接对比，表中已省略。",
        ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {"aggregate": agg, "results": results, "m1_results": m1_results or []},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return str(md_path), str(json_path)


def run_eval(
    n: int | None = 5,
    all_: bool = False,
    compare: bool = False,
    log=print,
) -> tuple[str, str]:
    """入口：跑评测并产出报告。返回 (report_md, results_json) 路径。"""
    topics = pick_topics(n=n, all_=all_)
    log(f"评测样本：{len(topics)} 条主题")

    results = [eval_one_m2(item, log=log) for item in topics]

    m1_results = None
    if compare:
        m1_results = [eval_one_m1(item, log=log) for item in topics]

    agg = aggregate(results)
    md_path, json_path = write_report(results, agg, m1_results)
    log(f"\n评测完成：{md_path}\n明细数据：{json_path}")
    return md_path, json_path
