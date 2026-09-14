# 分类计数 Agent 一键运行（PowerShell）
# 用法：解压到任意目录后，在本目录执行  .\run_counting_E.ps1
# 前提：
#   1) 已安装 python 3.12+ 与 uv（首次会自动同步依赖）
#   2) 仿真客户端已启动（E:\Windows\Run.bat）
#   3) 赛题系统已启动: start_test.bat train --task competition-preliminary-counting-task
#   4) 已设置大模型 Key:  $env:VLM_CLIENT_CFG_API_KEY = "你的key"

$ErrorActionPreference = 'Stop'
if (-not $env:VLM_CLIENT_CFG_API_KEY) {
    Write-Host "请先执行  `$env:VLM_CLIENT_CFG_API_KEY = '<你的大模型API Key>'  再运行本脚本" -ForegroundColor Yellow
    exit 1
}

$env:VLM_CLIENT_TYPE = "openai"
$env:VLM_CLIENT_CFG_NAME = "qwen3.7-flash"
$env:VLM_CLIENT_CFG_API_BASE = "https://tokenrhythm.studio/v1"

Set-Location $PSScriptRoot
Remove-Item ".counting_inventory.json" -ErrorAction SilentlyContinue
uv run python run_agent.py --agent_name counting_agent --config config.toml --run_times 12
