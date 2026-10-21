"""可观测追踪：一次调研运行 = 一份 trace 文件（outputs/traces/）。

记录四类事件：
- node    ：图节点 / 流程步骤的耗时与关键产物（如 RAG 统计、评审结论）
- llm     ：每次 LLM 调用的模型、延迟、token 用量、重试次数、错误
- event   ：运行级事件（运行开始、结束、状态）

使用方式（进程内全局单例，支持顺序多轮运行）：
    tracing.start_run(topic="...", mode="m2")
    with tracing.span("planner", {"queries": [...]}) as s:
        ...  # 节点工作；s.set(...) 可补充数据
    tracing.record_llm_call(...)
    summary = tracing.end_run(status="ok")

Langfuse 上报（可选）：.env 里同时配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
后，end_run 时会把 trace 与各 LLM 调用通过 Ingestion API 上报（httpx 直连，
不装 SDK）。上报失败只告警不阻塞——可观测本身不能成为新的故障点。
"""
import base64
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from src.config import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    TRACE_DIR,
    TRACE_ENABLED,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunTracer:
    """单次运行的结构化记录器。"""

    def __init__(self, topic: str, mode: str):
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.topic = topic
        self.mode = mode  # "m2" / "m1" / "eval"
        self.started_at = _utcnow()
        self.t0 = time.time()
        self.events: list[dict] = []
        self.current_node: str = ""
        # LLM 汇总（逐次调用也会进 events）
        self.llm_calls = 0
        self.llm_errors = 0
        self.llm_retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ---- 事件记录 ----

    def add_event(self, kind: str, name: str, data: dict | None = None) -> None:
        self.events.append(
            {"kind": kind, "name": name, "ts": _utcnow(), "data": data or {}}
        )

    def record_llm_call(
        self,
        *,
        prompt: str,
        output: str,
        latency_ms: float,
        model: str,
        attempts: int = 0,
        usage: dict | None = None,
        error: str | None = None,
    ) -> None:
        usage = usage or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        self.llm_calls += 1
        if error:
            self.llm_errors += 1
        if attempts:
            self.llm_retries += attempts
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.events.append(
            {
                "kind": "llm",
                "name": self.current_node or "llm",
                "ts": _utcnow(),
                "data": {
                    "model": model,
                    "latency_ms": round(latency_ms, 1),
                    "attempts": attempts,
                    "prompt_chars": len(prompt),
                    "output_chars": len(output or ""),
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "error": error,
                },
            }
        )

    # ---- 汇总与落盘 ----

    def summary(self, status: str = "ok", extra: dict | None = None) -> dict:
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "mode": self.mode,
            "status": status,
            "started_at": self.started_at,
            "duration_s": round(time.time() - self.t0, 1),
            "llm": {
                "calls": self.llm_calls,
                "errors": self.llm_errors,
                "retries": self.llm_retries,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
            "extra": extra or {},
        }

    def dump(self, status: str = "ok", extra: dict | None = None) -> dict:
        doc = {
            **self.summary(status, extra),
            "events": self.events,
        }
        if TRACE_ENABLED:
            try:
                TRACE_DIR.mkdir(parents=True, exist_ok=True)
                path = TRACE_DIR / f"{self.run_id}.json"
                path.write_text(
                    _json_dumps(doc), encoding="utf-8"
                )
                doc["_trace_file"] = str(path)
            except Exception as e:  # 落盘失败不影响主流程
                _warn(f"trace 落盘失败：{e}")
        return doc


def _json_dumps(obj: Any) -> str:
    import json

    def _default(o):
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    return json.dumps(obj, ensure_ascii=False, indent=2, default=_default)


def _warn(msg: str) -> None:
    print(f"[trace] {msg}")


# ===== 进程级全局：当前运行 =====

_current: RunTracer | None = None


def start_run(topic: str, mode: str = "m2") -> RunTracer:
    """开始一次运行追踪（上一轮未结束时先自动收尾为 interrupted）。"""
    global _current
    if _current is not None:
        finish("interrupted")
    _current = RunTracer(topic, mode)
    _current.add_event("event", "run_start", {"topic": topic, "mode": mode})
    return _current


def current() -> RunTracer | None:
    return _current


@contextmanager
def span(name: str, data: dict | None = None):
    """节点级计时区间：记录耗时与关键产物，并把节点名挂到 LLM 调用上。"""
    global _current
    tracer = _current
    if tracer is None:  # 未开启追踪（如单元测试）时零开销直通
        yield _NullSpan()
        return
    prev = tracer.current_node
    tracer.current_node = name
    t0 = time.time()
    s = _SpanData()
    try:
        yield s
    finally:
        tracer.current_node = prev
        payload = dict(data or {})
        payload.update(s.data)
        payload["duration_ms"] = round((time.time() - t0) * 1000, 1)
        tracer.add_event("node", name, payload)


class _SpanData:
    """span 内补充数据的小容器。"""

    def __init__(self) -> None:
        self.data: dict = {}

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


class _NullSpan:
    def set(self, key: str, value: Any) -> None:  # 未追踪时的空实现
        pass


def record_llm_call(**kwargs) -> None:
    if _current is not None:
        _current.record_llm_call(**kwargs)


def finish(status: str = "ok", extra: dict | None = None) -> dict | None:
    """收尾：写 trace 文件，返回汇总（含 Langfuse 上报结果）。"""
    global _current
    if _current is None:
        return None
    tracer = _current
    doc = tracer.dump(status, extra)
    doc["langfuse"] = _export_langfuse(tracer, doc)
    _current = None
    return doc


# ===== Langfuse 上报（可选，key 触发）=====

_LANGFUSE_MAX_CHARS = 2000  # 上报内容截断，避免 payload 过大


def _truncate(text: str, limit: int = _LANGFUSE_MAX_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"...(截断，共 {len(text)} 字符)"


def _export_langfuse(tracer: RunTracer, doc: dict) -> dict:
    """通过 Ingestion API 上报 trace + LLM generations。未配置 key 时返回 skipped。"""
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        return {"status": "skipped", "reason": "未配置 LANGFUSE_* key（本地 trace 照常工作）"}

    import httpx

    batch: list[dict] = [
        {
            "type": "trace-create",
            "body": {
                "id": tracer.run_id,
                "name": f"deep-research: {tracer.topic[:50]}",
                "timestamp": tracer.started_at,
                "metadata": {
                    "mode": tracer.mode,
                    "status": doc.get("status"),
                    "duration_s": doc.get("duration_s"),
                    "extra": doc.get("extra") or {},
                },
            },
        }
    ]
    for i, ev in enumerate(
        [e for e in tracer.events if e["kind"] == "llm"], start=1
    ):
        d = ev["data"]
        batch.append(
            {
                "type": "generation-create",
                "body": {
                    "id": f"{tracer.run_id}-gen{i}",
                    "traceId": tracer.run_id,
                    "name": ev["name"],
                    "startTime": ev["ts"],
                    "endTime": ev["ts"],
                    "model": d.get("model"),
                    "usage": {
                        "prompt": d.get("prompt_tokens") or 0,
                        "completion": d.get("completion_tokens") or 0,
                        "total": (d.get("prompt_tokens") or 0)
                        + (d.get("completion_tokens") or 0),
                    },
                    "metadata": {
                        "latency_ms": d.get("latency_ms"),
                        "attempts": d.get("attempts"),
                        "error": d.get("error"),
                    },
                },
            }
        )
    # 节点耗时作为 span 上报（只有起止时刻可还原，没有原始输入输出）
    for i, ev in enumerate(
        [e for e in tracer.events if e["kind"] == "node"], start=1
    ):
        dur = ev["data"].get("duration_ms")
        batch.append(
            {
                "type": "span-create",
                "body": {
                    "id": f"{tracer.run_id}-span{i}",
                    "traceId": tracer.run_id,
                    "name": ev["name"],
                    "startTime": ev["ts"],
                    "endTime": ev["ts"],
                    "metadata": ev["data"],
                }
            }
        )
        if dur is not None:
            batch[-1]["body"]["latency"] = dur

    try:
        auth = base64.b64encode(
            f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
        ).decode()
        resp = httpx.post(
            f"{LANGFUSE_HOST.rstrip('/')}/api/public/ingestion",
            json={"batch": batch},
            headers={"Authorization": f"Basic {auth}"},
            timeout=15,
        )
        if 200 <= resp.status_code < 300:
            return {"status": "ok", "events": len(batch)}
        return {
            "status": "error",
            "http_status": resp.status_code,
            "body": _truncate(resp.text, 500),
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
