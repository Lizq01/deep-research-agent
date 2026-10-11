"""深度调研 Agent（Deep Research）· M1 命令行入口

用法：
    python main.py "调研主题"        # 运行一次调研
    python main.py --check           # 检查运行环境（依赖 / 配置）

示例：
    python main.py "2026 年 AI Agent 行业发展现状"
"""
import sys
import traceback

from rich.console import Console
from rich.markdown import Markdown

from src.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, TAVILY_API_KEY

console = Console()


def check_env() -> bool:
    """检查依赖与配置，返回是否全部就绪。"""
    ok = True

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

    return ok


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

    topic = " ".join(args)

    try:
        from src.pipeline import run

        path = run(topic, log=lambda msg: console.print(msg))
    except Exception:
        console.print("[red]运行失败，完整错误如下：[/red]")
        traceback.print_exc()
        return

    console.print("\n[bold]===== 调研简报（预览）=====[/bold]\n")
    with open(path, encoding="utf-8") as f:
        console.print(Markdown(f.read()))


if __name__ == "__main__":
    main()
