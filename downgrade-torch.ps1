# ============================================================
# GPT-SoVITS torch 降级脚本 (浮浮酱定制)
# 目的: 把 torch 2.11.0+cu128 (与老训练代码错配, 反向时 0xC0000005 崩溃)
#       降到社区验证稳定的 torch 2.6.0 + torchaudio 2.6.0 + cu126
# 副作用红利: torchaudio<2.8 不再用 torchcodec 后端, 顺带消除 ffmpeg DLL 坑
# 用法: pwsh -ExecutionPolicy Bypass -File downgrade-torch.ps1
# 幂等: 已是目标版本则跳过安装
# ============================================================
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# 1) 注入 F 盘缓存变量 (禁写 C 盘)
. "$Root\_env.ps1"

# 2) 定位 venv 与目标版本
$VenvPy = "$Root\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "找不到 venv: $VenvPy" }
$env:VIRTUAL_ENV = "$Root\.venv"
$TargetTorch = "2.6.0"
$TargetAudio = "2.6.0"
$IndexUrl    = "https://download.pytorch.org/whl/cu124"

# 3) 检查当前版本, 已达标则跳过
$cur = & $VenvPy -c "import torch; print(torch.__version__)" 2>$null
Write-Host "[INFO] 当前 torch = $cur" -ForegroundColor Cyan
if ($cur -eq "$TargetTorch+cu124") {
    Write-Host "[SKIP] 已是目标版本, 无需重装" -ForegroundColor Green
} else {
    # 4) 卸载旧 torch 全家桶 + torchcodec (绑定 2.11, 必须移除)
    Write-Host "[STEP] 卸载 torch / torchaudio / torchcodec ..." -ForegroundColor Yellow
    uv pip uninstall --python $VenvPy torch torchaudio torchcodec 2>$null

    # 5) 安装目标版本 (大包 ~2.5G, 走 F 盘 TEMP 解压)
    Write-Host "[STEP] 安装 torch==$TargetTorch + torchaudio==$TargetAudio (cu126) ..." -ForegroundColor Yellow
    uv pip install --python $VenvPy `
        "torch==$TargetTorch" "torchaudio==$TargetAudio" `
        --index-url $IndexUrl
}

# 6) 验证版本
Write-Host "`n===== 验证安装结果 =====" -ForegroundColor Cyan
& $VenvPy -c "import torch,torchaudio; print('torch     :',torch.__version__); print('torchaudio:',torchaudio.__version__); print('cuda      :',torch.version.cuda); print('cudnn     :',torch.backends.cudnn.version()); print('gpu       :',torch.cuda.get_device_name(0))"

# 7) 最小 CUDA 反向测试 (复现原崩溃路径: conv1d + 上采样 convT1d + fp16)
Write-Host "`n===== 最小算子反向测试 (模拟训练崩溃点) =====" -ForegroundColor Cyan
& $VenvPy -c @"
import torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
x=torch.randn(8,1,16384,device='cuda',requires_grad=True)
c=nn.Conv1d(1,16,5,padding=2).cuda()
(c(x).sum()).backward(); torch.cuda.synchronize(); print('[OK] fp32 conv1d backward')
s=GradScaler()
with autocast():
    y=c(torch.randn(8,1,16384,device='cuda')).sum()
s.scale(y).backward(); torch.cuda.synchronize(); print('[OK] fp16 autocast conv1d backward')
ct=nn.ConvTranspose1d(16,8,16,stride=8,padding=4).cuda()
z=torch.randn(8,16,2048,device='cuda',requires_grad=True)
(ct(z).sum()).backward(); torch.cuda.synchronize(); print('[OK] convT1d backward')
print('[PASS] 所有最小反向测试通过, 底层环境健康')
"@

Write-Host "`n[DONE] 降级完成喵~ 接下来回 webui 重新点训练即可" -ForegroundColor Green
