"""config 层：所有参数与环境变量的唯一出口。

settings 负责从 .env 读取并做类型转换；这里整体转发，
让上层继续写 `from src.config import XXX` 而不用关心内部分层。
"""
from src.config.settings import *  # noqa: F401,F403
