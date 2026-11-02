"""MCP Client：用 stdio 连接本地 MCP Server，把它的工具接进 LangGraph。

为什么是「常驻会话」而不是每次现连：
- LangGraph 的节点是同步函数，一次调研里多个 Researcher 并发跑在不同线程；
- 每调用一次就起一个子进程，意味着重新加载 FAISS 索引、BM25 语料和嵌入模型（1~2s 冷启动），
  4 个子问题就是 4 次冷启动；
- 所以这里起一条后台事件循环线程，握手一次、会话常驻，之后每次调用只是一次 JSON-RPC 往返。

线程模型：
- 主线程（LangGraph 节点）调用同步方法 → `run_coroutine_threadsafe` 把协程丢进后台循环 → 阻塞取结果；
- 多个线程共享同一个 session 是合法的：JSON-RPC 请求带 id，响应由 SDK 的 dispatcher 分发；
- 子进程生命周期由 SDK 负责（stdin 关闭 → 优雅退出 → 超时杀进程树），close() 显式触发。

凭据边界：
stdio 传输下，服务端只继承 SDK 的白名单环境变量（DEFAULT_INHERITED_ENV_VARS），
Client 侧额外通过 env 显式指定；本模块传的 env 里没有任何 KEY/TOKEN，
服务端凭据由它自己读 .env —— 换句话说，Client 想泄漏也没有途径。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    """连接/会话级失败：起不来、断连、超时。"""


class MCPToolError(RuntimeError):
    """工具执行失败：服务端返回 isError（参数校验不通过、chunk 不存在等）。"""

    def __init__(self, tool: str, message: str):
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.detail = message


@dataclass(frozen=True)
class ToolSpec:
    """从 tools/list 拿到的工具描述。

    input_schema 是服务端给的 JSON Schema —— 调用方可以据此只传服务端确实支持的参数，
    而不是按硬编码的假设传参（这正是动态发现的意义）。
    """

    name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    @property
    def params(self) -> tuple[str, ...]:
        return tuple((self.input_schema.get("properties") or {}).keys())

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.input_schema.get("required") or [])

    def describe(self) -> str:
        return f"{self.name}({', '.join(self.required) or '无必填'})"


class MCPToolClient:
    """同步外观 + 异步内核的 MCP 客户端。"""

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        connect_timeout: float = 30.0,
        call_timeout: float = 60.0,
        log_path: str | Path | None = None,
        inherit_stderr: bool = False,
    ):
        self._params = StdioServerParameters(
            command=command,
            args=list(args),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
        )
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._log_path = Path(log_path) if log_path else None
        self._inherit_stderr = inherit_stderr

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._tools: list[ToolSpec] = []
        self.server_info: dict = {}

        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._lifecycle_lock = threading.Lock()

    # ------------------------------------------------------------ 生命周期

    def start(self) -> None:
        """建立会话（幂等）。失败抛 MCPClientError，调用方决定降级还是中断。

        多线程并发首次调用时，只有一条线程负责拉起子进程，其余线程在 _ready 上等同一份结果；
        因此「线程已存在」也必须等 ready，不能直接返回。
        """
        with self._lifecycle_lock:
            alive = self._thread is not None and self._thread.is_alive()
            if not alive:
                self._ready.clear()
                self._start_error = None
                self._thread = threading.Thread(
                    target=self._loop_main, name="mcp-client", daemon=True)
                self._thread.start()

        if not self._ready.wait(self._connect_timeout):
            raise MCPClientError(
                f"连接 MCP Server 超时（{self._connect_timeout:.0f}s）：{self._describe()}")
        if self._start_error is not None:
            raise MCPClientError(
                f"连接 MCP Server 失败：{type(self._start_error).__name__}: "
                f"{self._start_error}（{self._describe()}）") from self._start_error

    def close(self, timeout: float = 10.0) -> None:
        with self._lifecycle_lock:
            loop, thread, stop_event = self._loop, self._thread, self._stop_event
            self._thread = None
            self._loop = None
            self._session = None
            self._stop_event = None
        if loop is None or thread is None or not thread.is_alive():
            return
        try:
            if stop_event is not None:
                # 会话协程退出 → stdio_client 的上下文收尾 → 子进程被终止
                loop.call_soon_threadsafe(stop_event.set)
            else:
                loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:  # loop 已关闭
            return
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("MCP Client 线程未在 %.0fs 内退出，子进程可能残留", timeout)

    def _describe(self) -> str:
        return f"{self._params.command} {' '.join(self._params.args)} (cwd={self._params.cwd})"

    def _loop_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001 —— 分启动期/运行期两种失败
            if self._ready.is_set():
                logger.warning("MCP 会话异常结束：%s: %s", type(exc).__name__, exc)
            else:
                self._start_error = exc
        finally:
            self._ready.set()  # 无论如何都放行 start()，避免调用方干等
            self._session = None
            loop.close()

    async def _serve(self) -> None:
        """连接 → 握手 → 保持会话，直到 close() 触发。

        注意：stdio_client / ClientSession 的上下文必须在**同一个协程任务**里进出。
        anyio 的 cancel scope 绑定任务，若把 connect 和 aclose 拆成两次
        run_until_complete（两个任务），关会话时会抛
        "Attempted to exit cancel scope in a different task than it was entered in"。
        """
        self._stop_event = asyncio.Event()
        errlog = self._open_errlog()
        try:
            async with stdio_client(self._params, errlog=errlog) as (read, write):
                async with ClientSession(
                    read, write, read_timeout_seconds=self._call_timeout
                ) as session:
                    init = await session.initialize()
                    self._session = session
                    self.server_info = {
                        "name": init.server_info.name,
                        "version": init.server_info.version,
                        "protocol": init.protocol_version,
                    }
                    self._tools = await self._fetch_tools()
                    self._ready.set()
                    await self._stop_event.wait()
        finally:
            if errlog is not sys.stderr:
                errlog.close()

    def _open_errlog(self):
        """服务端 stderr 的去处。

        默认写文件而不是直接打屏：stdio 下 stdout 是协议流、stderr 是日志流，
        把日志混进终端输出会盖掉 Agent 自己的进度信息；排查时看 outputs/mcp_server.log。
        """
        if self._inherit_stderr or self._log_path is None:
            return sys.stderr
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            return self._log_path.open("a", encoding="utf-8")
        except OSError:
            return sys.stderr

    # ------------------------------------------------------------ 同步接口

    def list_tools(self, refresh: bool = False) -> list[ToolSpec]:
        """拉取工具列表（默认走缓存，refresh=True 强制重新 tools/list）。"""
        self.start()
        if refresh or not self._tools:
            self._tools = self._run(self._fetch_tools())
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        """调用工具，返回结构化解包后的 dict。

        服务端返回 isError 时抛 MCPToolError（含服务端给的原始错误信息，
        例如参数校验失败的字段与取值范围），便于上层决定「换参数重试」还是「跳过」。
        """
        self.start()
        result = self._run(self._call_tool_async(name, arguments or {}))
        return self._decode(name, result)

    def has_tool(self, name: str) -> bool:
        return any(t.name == name for t in self.list_tools())

    # ------------------------------------------------------------ 协程实现

    async def _fetch_tools(self) -> list[ToolSpec]:
        assert self._session is not None
        result = await self._session.list_tools()
        if getattr(result, "next_cursor", None):
            # 工具少于一页时不会发生；真发生了说明服务端工具很多，需要分页拉全
            logger.warning("tools/list 返回分页游标，当前只取了第一页")
        specs = []
        for tool in result.tools:
            schema = dict(tool.input_schema or {})
            specs.append(ToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=schema,
            ))
        return specs

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]):
        assert self._session is not None
        return await self._session.call_tool(name, arguments)

    def _run(self, coro):
        loop = self._loop
        if loop is None or self._session is None:
            raise MCPClientError("MCP 会话未建立")
        if self._thread is not None and not self._thread.is_alive():
            raise MCPClientError("MCP Client 线程已退出，会话不可用")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(self._call_timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise MCPClientError(f"MCP 调用超时（{self._call_timeout:.0f}s）") from exc

    @staticmethod
    def _decode(name: str, result) -> dict:
        text = "".join(
            getattr(item, "text", "") for item in (result.content or [])
        )
        if getattr(result, "is_error", False):
            raise MCPToolError(name, text.strip()[:800] or "工具执行失败（无错误信息）")
        if result.structured_content is not None:
            return result.structured_content
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
        return payload if isinstance(payload, dict) else {"data": payload}
