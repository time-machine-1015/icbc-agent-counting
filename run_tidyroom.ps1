# 整理房间任务一键启动脚本（PowerShell）
# 用法: 在 baseline-agent-master 目录下执行  .\run_tidyroom.ps1
# 前提: UE 客户端(E:\Windows\Run.bat) 与 赛题系统(start_test.bat train/test --task competition-preliminary-tidy-room-task) 已启动

# ---- VLM 配置（OpenAI 兼容中转）----
# 密钥放环境变量，勿写进脚本。运行前: $env:VLM_CLIENT_CFG_API_KEY = "你的key"
$env:VLM_CLIENT_TYPE = "openai"
$env:VLM_CLIENT_CFG_NAME = "qwen3.7-flash"
$env:VLM_CLIENT_CFG_API_BASE = "https://tokenrhythm.studio/v1"
if (-not $env:VLM_CLIENT_CFG_API_KEY) {
    Write-Error "请先设置环境变量 VLM_CLIENT_CFG_API_KEY（大模型 API Key）后重试。"
    exit 1
}

uv run python run_agent.py --agent_name tidy_room_agent --config config.toml --run_times 5
