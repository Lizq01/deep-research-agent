"""深度调研 Agent（Deep Research）· 命令行入口（M2）

用法：
    python main.py "调研主题"        # M2：LangGraph 多智能体（默认）
    python main.py --yes "调研主题"  # 同上，人工审核自动通过（非交互 / 脚本用）
    python main.py --m1 "调研主题"   # M1：顺序流程（对照用）
    python main.py --check           # 检查运行环境（依赖 / 配置）

示例：
    python main.py "2026 年 AI Agent 行业发展现状"
"""
import sys
import traceback
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown

from src.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, TAVILY_API_KEY

console = Console()


def check_env(connectivity: bool = True) -> bool:
    """检查依赖与配置，返回是否全部就绪。

    connectivity=True 时额外做真实连通性测试（1 次极短 LLM 调用 + 1 次搜索）。
    """
    ok = True

    console.print("[bold]0) 运行时[/bold]")
    console.print(f"   Python：{sys.executable}")
    if ".venv" not in sys.executable.replace("\\", "/"):
        console.print(
            "   [red]警告[/red]  当前不是项目 .venv 里的 Python！"
            "这可能就是「用不了」的原因。"
        )
        console.print(
            "   请用 [bold].venv\\Scripts\\python.exe main.py --check[/bold] "
            "或在 PyCharm 里把解释器切到 .venv\\Scripts\\python.exe"
        )
        ok = False

    console.print("[bold]1) 依赖检查[/bold]")
    for mod in ["openai", "dotenv", "httpx", "bs4", "rich", "tavily", "ddgs", "langgraph"]:
        try:
            __import__(mod)
            console.print(f"   [green]OK[/green]    {mod}")
        except Exception as e:
            ok = False
            console.print(f"   [red]缺失[/red]  {mod}  ({e})")

    console.print("[bold]2) 配置检查（.env）[/bold]")
    if OPENAI_API_KEY:
        console.print("   [green]OK[/green]    OPENAI_API_KEY 已配置")
        console.print(f"   [green]OK[/green]    MODEL_NAME = {MODEL_NAME}")
        console.print(f"   [green]OK[/green]    OPENAI_BASE_URL = {OPENAI_BASE_URL or '(默认 OpenAI 官方)'}")
    else:
        ok = False
        console.print("   [red]缺失[/red]  未找到 OPENAI_API_KEY —— 请把 .env.example 复制为 .env 并填写")

    if TAVILY_API_KEY:
        console.print("   [green]OK[/green]    TAVILY_API_KEY 已配置（搜索走 Tavily）")
    else:
        console.print("   [yellow]提示[/yellow]  未配置 TAVILY_API_KEY，将使用免费的 DuckDuckGo 搜索")

    if connectivity:
        console.print("[bold]3) 连通性测试（真实调用，会消耗极少量额度）[/bold]")
        if not OPENAI_API_KEY:
            console.print("   [yellow]跳过[/yellow]  未配置 OPENAI_API_KEY")
        else:
            try:
                from src.llm import chat

                out = chat("只回复两个字：收到", temperature=0.0)
                console.print(f"   [green]OK[/green]    LLM 连通（应答：{out[:20]}）")
            except Exception as e:
                ok = False
                console.print(f"   [red]失败[/red]  LLM 调用异常：{str(e)[:200]}")

        try:
            from src.tools import web_search

            rows = web_search("test", max_results=1)
            if rows:
                console.print(
                    f"   [green]OK[/green]    搜索连通（{len(rows)} 条，"
                    f"通道：{'Tavily' if TAVILY_API_KEY else 'DuckDuckGo'}）"
                )
            else:
                console.print(
                    "   [yellow]警告[/yellow]  搜索返回 0 条 —— 检索会拿不到资料。"
                    "建议配置免费的 TAVILY_API_KEY（https://tavily.com）"
                )
        except Exception as e:
            console.print(f"   [yellow]警告[/yellow]  搜索异常：{str(e)[:200]}")

    return ok


def run_m1(topic: str) -> str:
    from src.pipeline import run

    return run(topic, log=lambda msg: console.print(msg))


def run_m2(topic: str, auto_approve: bool = False) -> str | None:
    """LangGraph 多智能体流程，含 interrupt 人工审核交互。

    auto_approve=True 时跳过人工闸门（非交互环境 / 脚本），审核结论仍会打印。
    """
    from langgraph.types import Command

    from src.graph import build_graph

    graph = build_graph()
    thread_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # recursion_limit：有界自主的兜底闸门，任何异常循环都会被强制终止
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    result = graph.invoke({"topic": topic}, config=config)

    while interrupts := result.get("__interrupt__"):
        payload = interrupts[0].value
        review = payload.get("review", {})

        console.print("\n[bold yellow]===== 人工审核（报告尚未落盘）=====[/bold yellow]")
        console.print(
            f"评审得分：{review.get('score', 'N/A')}/10"
            f"（修订 {payload.get('revision_count', 0) - 1} 次）"
        )
        console.print("\n[bold]--- 简报预览 ---[/bold]\n")
        console.print(Markdown(payload.get("report", "")))

        if auto_approve:
            console.print("\n[yellow]--yes 已开启，自动通过审核。[/yellow]")
            resume = {"action": "approve"}
        else:
            resume = _ask_decision()

        result = graph.invoke(Command(resume=resume), config=config)

    if result.get("aborted"):
        console.print("\n[yellow]已放弃，报告未保存。[/yellow]")
        return None
    return result.get("final_path")


def _ask_decision() -> dict:
    """读取人工审核决策：y 通过 / 修改意见退回 / n 放弃。

    空输入会重新提示；stdin 不可用（EOF / 非交互）时按放弃处理，避免静默卡死。
    """
    for _ in range(3):
        try:
            ans = console.input(
                "\n[bold]输入 y 通过并保存 / 直接输入修改意见退回修订 / n 放弃：[/bold]"
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print(
                "\n[yellow]未读取到输入（非交互环境）。如要跳过审核，"
                "请加 --yes 参数重跑。[/yellow]"
            )
            return {"action": "abort"}

        if not ans:
            console.print("[yellow]没有输入内容，请重新输入。[/yellow]")
            continue
        if ans.lower() == "y":
            return {"action": "approve"}
        if ans.lower() == "n":
            return {"action": "abort"}
        return {"action": "revise", "comment": ans}

    console.print("[yellow]连续空输入，按放弃处理。[/yellow]")
    return {"action": "abort"}


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in {"-h", "--help"}:
        console.print(__doc__)
        return

    if args[0] == "--check":
        ok = check_env()
        if ok:
            console.print("\n[bold green]环境检查通过，可以开始调研了。[/bold green]")
        else:
            console.print("\n[bold red]环境不完整，请按上面提示修复后重试。[/bold red]")
        return

    auto_approve = False
    use_m1 = False
    while args and args[0].startswith("--"):
        flag = args[0]
        if flag in {"--yes", "-y"}:
            auto_approve = True
        elif flag == "--m1":
            use_m1 = True
        else:
            console.print(f"[red]未知参数：{flag}[/red]")
            console.print(__doc__)
            return
        args = args[1:]

    if not args:
        console.print("[red]请提供调研主题，例如：python main.py \"2026 年 AI Agent 行业现状\"[/red]")
        console.print(__doc__)
        return

    topic = " ".join(args)

    if args and not topic.strip():
        console.print("[red]调研主题不能为空。[/red]")
        return

    try:
        path = run_m1(topic) if use_m1 else run_m2(topic, auto_approve=auto_approve)
    except Exception:
        console.print("[red]运行失败，完整错误如下：[/red]")
        traceback.print_exc()
        return

    if path:
        console.print(f"\n[bold green]完成，报告路径：{path}[/bold green]")


if __name__ == "__main__":
    main()
