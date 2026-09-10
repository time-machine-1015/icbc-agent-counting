# 分类计数任务一键启动脚本（PowerShell）— 主清单缓存 + run_times 多连接一次跑完 10 题
# 用法: 在 baseline-agent-master 目录下执行  .\run_counting.ps1
# 前提: UE 客户端(E:\Windows\run.bat) 与 赛题系统(start_test.bat train/test --task competition-preliminary-counting-task) 已启动

$ErrorActionPreference = 'SilentlyContinue'
# 每次整轮开始前删缓存，保证首连接必巡游建清单，其余连接复用
Remove-Item -LiteralPath ".counting_inventory.json" -ErrorAction SilentlyContinue

# ---- VLM 配置（OpenAI 兼容中转）----
# 密钥请放在环境变量里，不要写进脚本/提交。运行前先设置： $env:VLM_CLIENT_CFG_API_KEY = "你的key"
$env:VLM_CLIENT_TYPE = "openai"
$env:VLM_CLIENT_CFG_NAME = "qwen3.7-flash"
$env:VLM_CLIENT_CFG_API_BASE = "https://tokenrhythm.studio/v1"
if (-not $env:VLM_CLIENT_CFG_API_KEY) {
    Write-Error "请先设置环境变量 VLM_CLIENT_CFG_API_KEY（大模型 API Key）后重试。"
    exit 1
}

uv run python run_agent.py --agent_name counting_agent --config config.toml --run_times 12
