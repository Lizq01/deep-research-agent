"""Knowledge：经 MCP 调用内部知识库，产出与网页证据同构的 Evidence。

为什么放在 Researcher 之外单独成一个节点：
- 数据来源不同（内网知识库 vs 公网网页），失败模式也不同（MCP 连不上 vs 搜索被限流）；
- 分开之后，MCP 挂掉只是少一路证据，不会拖垮整条调研链路（降级契约）。

「动态发现」具体指三件事，而不是嘴上说说：
1. 每次运行先 tools/list，按**服务端实际返回**的工具名选工具，不假设一定叫 search_knowledge；
2. 按工具自己的 input_schema 过滤参数，服务端没声明的参数不传（改名/删参数不会直接报错）；
3. 按 required 校验必填项，缺了就在本地报错，而不是发一次注定被拒的请求。
"""
from __future__ import annotations

import atexit
import math
import time

from src.config import (
    MCP_CALL_TIMEOUT,
    MCP_CONNECT_TIMEOUT,
    MCP_ENABLED,
    MCP_MAX_EVIDENCE,
    MCP_MAX_QUERIES,
    MCP_SERVER_ARGS,
    MCP_SERVER_CWD,
    MCP_SERVER_LOG,
    MCP_STDERR_INHERIT,
    MCP_TOP_K,
    resolve_mcp_server_command,
)
from src.modules.mcp_client import (
    MCPClientError,
    MCPToolClient,
    MCPToolError,
    ToolSpec,
)
from src.utils.console import console as _console

_client: MCPToolClient | None = None
_client_error: str | None = None  # 缓存失败原因：连接失败不必每个子问题都再等一次超时


def get_client() -> MCPToolClient:
    """进程内单例：图里多个节点共享同一条 MCP 会话。"""
    global _client, _client_error
    if _client is not None:
        return _client
    if _client_error is not None:
        raise MCPClientError(_client_error)

    client = MCPToolClient(
        command=resolve_mcp_server_command(),
        args=MCP_SERVER_ARGS,
        cwd=MCP_SERVER_CWD,
        connect_timeout=MCP_CONNECT_TIMEOUT,
        call_timeout=MCP_CALL_TIMEOUT,
        log_path=MCP_SERVER_LOG,
        inherit_stderr=MCP_STDERR_INHERIT,
    )
    try:
        client.start()
        client.list_tools()
    except Exception as exc:  # noqa: BLE001 —— 失败要落成可读信息，且不要反复重连
        _client_error = f"{type(exc).__name__}: {exc}"
        client.close()
        raise MCPClientError(_client_error) from exc
    _client = client
    return client


def shutdown_client() -> None:
    """关闭会话并终止服务端子进程（进程退出前调用）。"""
    global _client, _client_error
    if _client is not None:
        _client.close()
    _client = None
    _client_error = None


def _drop_client() -> None:
    global _client, _client_error
    if _client is not None:
        _client.close()
    _client = None
    _client_error = "会话已断开"


def _pick_search_tool(tools: list[ToolSpec]) -> ToolSpec | None:
    """在服务端声明的工具里挑检索工具：优先精确名，其次按名字含 search 兜底。"""
    return (
        next((t for t in tools if t.name == "search_knowledge"), None)
        or next((t for t in tools if "search" in t.name.lower()), None)
    )


def collect_kb_evidence(queries: list[str], log=print) -> tuple[list[dict], dict]:
    """按子问题检索内部知识库，返回 ([Evidence], stats)。任何失败都不抛，降级为空证据。"""
    stats: dict = {"status": "ok", "tools": [], "calls": 0, "hits": 0, "errors": []}
    if not MCP_ENABLED:
        return [], {"status": "disabled"}

    started = time.perf_counter()
    try:
        client = get_client()
        tools = client.list_tools()          # 动态发现：以服务端返回为准
    except MCPClientError as exc:
        log(f"      MCP 不可用，跳过内部知识库：{exc}")
        return [], {**stats, "status": "unavailable", "errors": [str(exc)]}

    stats["tools"] = [t.name for t in tools]
    stats["server"] = client.server_info
    search = _pick_search_tool(tools)
    if search is None:
        msg = f"服务端未提供检索类工具，实际提供：{stats['tools']}"
        log(f"      {msg}")
        return [], {**stats, "status": "tool_missing", "errors": [msg]}

    log(f"      MCP 工具发现：{', '.join(t.describe() for t in tools)}")

    evidence: list[dict] = []
    seen: set[str] = set()
    # 均摊配额：否则第一个子问题就能吃满 MCP_MAX_EVIDENCE，后面的子问题一条都进不来
    picked_queries = queries[:MCP_MAX_QUERIES]
    quota = max(1, math.ceil(MCP_MAX_EVIDENCE / max(1, len(picked_queries))))

    for query in picked_queries:
        args: dict = {"query": query, "top_k": MCP_TOP_K}
        # 只传服务端 Schema 里声明过的参数：服务端改名/去掉 top_k 时仍能调用
        args = {k: v for k, v in args.items() if k in search.params}
        missing = [p for p in search.required if p not in args]
        if missing:
            msg = f"{search.name} 的必填参数 {missing} 无法从本地配置提供"
            log(f"      {msg}")
            stats["errors"].append(msg)
            break

        try:
            payload = client.call_tool(search.name, args)
        except MCPToolError as exc:
            # 单次调用被拒（参数校验、chunk 不存在）→ 换下一个子问题，不断连
            stats["errors"].append(str(exc))
            log(f"      MCP 调用被拒：{exc}")
            continue
        except MCPClientError as exc:
            # 会话级失败 → 断开，后面的子问题不必再等超时
            stats["errors"].append(str(exc))
            log(f"      MCP 会话异常，停止后续调用：{exc}")
            _drop_client()
            break

        stats["calls"] += 1
        results = payload.get("results") or []

        taken = 0
        for item in results:
            chunk_id = item.get("chunk_id")
            text = (item.get("text") or "").strip()
            if not chunk_id or not text or chunk_id in seen:
                continue
            seen.add(chunk_id)
            doc, position = item.get("doc", "unknown"), item.get("position", 0)
            evidence.append({
                "title": f"内部知识库《{doc}》第 {position} 段",
                # 用 mcp:// 伪 URL 保留出处：aggregator 按 URL 去重，报告里也能看出是内部资料
                "url": f"mcp://rag-knowledge/{doc}#{position}",
                "content": text,
                "query": query,
            })
            taken += 1
            if taken >= quota or len(evidence) >= MCP_MAX_EVIDENCE:
                break
        log(f"      · MCP {query} → 命中 {len(results)} 条，取 {taken} 条")
        if len(evidence) >= MCP_MAX_EVIDENCE:
            break

    stats["hits"] = len(evidence)
    stats["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return evidence, stats


def knowledge_node(state: dict) -> dict:
    """LangGraph 节点：并行于 Researcher 分支，为后续 RAG/Writer 补充内部资料。"""
    from src.modules import tracing

    queries = list(state.get("queries") or [])
    _console.print(f"[2/6] Knowledge 通过 MCP 查询内部知识库（{len(queries)} 个子问题）...")
    with tracing.span("knowledge") as span:
        evidence, stats = collect_kb_evidence(queries, log=_console.print)
        span.set("mcp_status", stats.get("status"))
        span.set("mcp_hits", stats.get("hits", 0))
        span.set("mcp_calls", stats.get("calls", 0))
    if evidence:
        _console.print(f"      内部知识库命中 {len(evidence)} 条片段")
    return {"kb_evidence": evidence, "mcp_stats": stats}


# 兜底：进程退出时关掉 MCP 会话，避免服务端子进程残留（显式调用过则此处为 no-op）
atexit.register(shutdown_client)
