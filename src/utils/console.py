"""共享的 Rich Console。

为什么单独抽出来：LangGraph 的 Send 分支会并行跑在多个线程里，
如果每个模块各自 new 一个 Console，它们各有各的锁，两次 print 的输出会互相插进对方行内。
全局共用一个实例，Rich 内部的锁才有意义。
"""
from rich.console import Console

console = Console()
