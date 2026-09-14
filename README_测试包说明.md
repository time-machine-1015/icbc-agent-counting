# 工行杯·分类计数 Agent（测试包说明）

## 这是什么
第十七届"工行杯"高校组智能体赛 · **分类计数任务（competition-preliminary-counting-task）** 的专用 Agent。
基于官方 baseline（ArenaAgentPro / tongsim gRPC）二次开发。

核心思路（对应赛题"越快答对分越高，答对才出分"）：
1. **每题重新感知**：出生点原地 3 张×120° 建初始清单 + 物体外接矩形(bbox)；
2. **4 点法巡游**（v2 提速）：
   - 由 bbox 推出房间四角，**排除"床角"**（清单里 shape/name 含 bed；认不出则用**最大件家具**代理，排除离它最近的角）；
   - 其余 3 个角：朝**房间中心**内缩 120cm + 可落点校验，到点后**只拍 1 张**（FOV 120° 已覆盖该角朝内的 90°）；
   - 第 4 点 = **房间正中心**：3 张×120° 全向观测；
   - 从出生点做最近邻串联，`move_to_location` 引擎寻路（自动绕墙/失败跳过/黑名单）；
   - **安全网**：4 点后清单 < 15 件 → 自动补看最大件家具背视点 1~2 个再答；
3. **两级作答**：颜色/类别词能在清单精确命中（且数量恰为某选项值）→ **代码直接数，毫秒级**；否则 **一次多模态调用**（2 张巡游视角图 + 带 id 分割图 + 清单），"以看图为准"识别 Unknown 物体的类别（碗/钟/瓶等），镜子按 id 判重；
4. **提交格式**：服务端要的是**选项数值整数**（实测 answer=5 → 选中值=5 的选项 G），答对才推进下一题；**每题服务端会结束连接，必须靠 `run_times≥10` 重连拿下一题**；
5. **失败兜底**：感知挂→快速退出（无超时幽灵）；重试换相位重巡；尝试耗尽→在未试过的选项里按估计值就近试探。

## 包内 / 包外
| 有 | 无（需自备） |
|---|---|
| `arenaagent/counting_agent/`（计数 agent+提示词）| 仿真客户端 UE（`Windows.zip` 解压的 E:\Windows）|
| `arenaagent/tidy_room_agent/`（整理房间 agent）| 赛题系统 `release`（arena_offline + tongsim_server）|
| 官方 baseline / 协议 / 启动脚本 / 离线测试 `tests_local/` | 大模型 API Key（环境变量注入，不落盘）|
| 说明文档 docs/ | `.venv`（用 uv sync 一键重建）|

## 快速开始（Windows）
```powershell
# 0) 装 python 3.12+ 和 uv（官网或 pip install uv）
# 1) 解压 → 进目录同步依赖
uv sync
# 2) 依次启动：UE 客户端 Run.bat → 赛题系统 start_test.bat train --task competition-preliminary-counting-task
# 3) 注入 Key 并运行本 Agent
$env:VLM_CLIENT_CFG_API_KEY = "<你的OpenAI兼容Key>"   # 默认走 tokenrhythm 中转+qwen3.7-flash，可改 run_counting_E.ps1 里 base/name
.\run_counting_E.ps1
```
- train 模式逐题结果在 `arena_offline/eval_res.json`（会被归档 `eval_res.<ts>.json`）
- 正式提交：`test` 模式跑完，`start_test.bat package-result` 打包上传

## 不依赖仿真器的离线自检（先跑这个验证代码没被改坏）
```powershell
uv run python tests_local/counting_offline_test.py   # 提交格式/竞态复用/缓存复用
uv run python tests_local/counting_codegate_test.py  # 代码门控:该秒答的绝不调模型
uv run python tests_local/counting_nav_test.py       # 路径规划:藏柜后的物能否被找到/不撞墙
```

## 实测成绩（本机 16GB/核显，UE 常崩的环境下）
单题崩溃前均**首答即对**：70.15 / 74.5 / 69.4 / 68.5 / 80.2 / 83.5 / 82.3 分档；
正确率瓶颈已解决（带图计数解决"碗/钟/瓶"类别盲区，影子观察点解决遮挡漏数），
剩余瓶颈=机器内存（UE5 每 5-15 分钟崩一次），建议 16G+独显 或 32G 内存机长跑。

## 主要参数（config.toml [counting_agent]）
`roam_time_budget_s`(首巡预算) `roam_max_points` `sweep_views`(每点视角)
`max_answer_images`(喂几张图) `max_attempts_per_subject`(重试后转选项试探)
`run_budget_s` 与题号推进宽限 `advance_wait_s`。

代码入口：`arenaagent/counting_agent/counting_agent.py`（run() 覆盖基类：逐题重感知+试探兜底）
