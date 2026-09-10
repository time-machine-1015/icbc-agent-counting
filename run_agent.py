#!/usr/bin/env python
"""Agent 启动器。

用 `uv run python run_agent.py ...` 代替 `uv run arenaagent ...`：
- 控制台脚本入口不会把当前目录加入 sys.path，导致生成的 pb2 依赖的
  顶层 arena 兼容包不可见；
- `python -m arenaagent.builder` 又会把 builder 以 __main__ 与
  arenaagent.builder 两个身份导入，@Register 注册表分裂。
本脚本以普通模块方式调用 arenaagent.builder.main()，两个问题都不存在。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arenaagent.builder import main

if __name__ == "__main__":
    main()
