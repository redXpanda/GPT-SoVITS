# ============================================================
# GPT-SoVITS 本地一键安装脚本 (浮浮酱定制)
#   特性: uv+.venv (无需conda) / 全缓存重定向F盘(禁写C盘) / 幂等可重跑
#   用法: pwsh -F setup-local.ps1 [-Device CU128] [-Source HF-Mirror] [-NoUVR5]
# ============================================================
Param(
    [ValidateSet("CU128", "CU126", "CPU")][string]$Device = "CU128",
    [ValidateSet("HF", "HF-Mirror", "ModelScope")][string]$Source = "HF-Mirror",
    [string]$CacheRoot = "F:\ai\cache",
    [string]$PyVersion = "3.10",
    [switch]$NoUVR5
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

function Info($m)    { Write-Host "[INFO] "    -ForegroundColor Green     -NoNewline; Write-Host $m }
function Step($m)    { Write-Host "`n==== $m ====" -ForegroundColor Cyan }
function Success($m) { Write-Host "[OK] "      -ForegroundColor Blue      -NoNewline; Write-Host $m }
function Warn($m)    { Write-Host "[WARN] "    -ForegroundColor Yellow    -NoNewline; Write-Host $m }

$Py = Join-Path $ScriptDir ".venv\Scripts\python.exe"

# ---- 模型下载源 ----
switch ($Source) {
    "HF"         { $Base = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" }
    "HF-Mirror"  { $Base = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" }
    "ModelScope" { $Base = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master" }
}
$HFEndpoint = if ($Source -eq "HF-Mirror") { "https://hf-mirror.com" } else { "https://huggingface.co" }

# ============================================================
# Step 1: 创建缓存目录结构 (全部落F盘)
# ============================================================
Step "Step 1/9  创建缓存目录 $CacheRoot"
$SubDirs = "uv", "uv-python", "tmp", "huggingface", "modelscope", "torch", "nltk_data", "pip"
foreach ($d in $SubDirs) { New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot $d) | Out-Null }
Success "缓存目录就绪"

# ============================================================
# Step 2: 注入环境变量 (本进程, 劫持所有缓存到F盘)
# ============================================================
Step "Step 2/9  注入缓存环境变量 (禁写C盘)"
$env:UV_CACHE_DIR          = "$CacheRoot\uv"
$env:UV_PYTHON_INSTALL_DIR = "$CacheRoot\uv-python"
$env:TMP                   = "$CacheRoot\tmp"
$env:TEMP                  = "$CacheRoot\tmp"
$env:PIP_CACHE_DIR         = "$CacheRoot\pip"
$env:HF_HOME               = "$CacheRoot\huggingface"
$env:HF_ENDPOINT           = $HFEndpoint
$env:MODELSCOPE_CACHE      = "$CacheRoot\modelscope"
$env:TORCH_HOME            = "$CacheRoot\torch"
$env:NLTK_DATA             = "$CacheRoot\nltk_data"
Success "环境变量已注入 -> $CacheRoot"

# ============================================================
# Step 3: 用 uv 创建 .venv (解释器也落F盘)
# ============================================================
Step "Step 3/9  创建虚拟环境 (.venv, Python $PyVersion)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "未找到 uv, 请先安装 uv (https://docs.astral.sh/uv/)" }
if (Test-Path $Py) {
    Info "已存在 .venv, 跳过创建"
} else {
    & uv venv --python $PyVersion .venv
    if ($LASTEXITCODE -ne 0) { throw "uv venv 创建失败" }
}
Success "虚拟环境就绪: $Py"

# ============================================================
# Step 4: 安装 PyTorch (跳过 torchcodec: Windows无wheel)
# ============================================================
Step "Step 4/9  安装 PyTorch ($Device)"
& $Py -c "import torch" 2>$null
$torchInstalled = ($LASTEXITCODE -eq 0)
if ($torchInstalled) {
    Info "torch 已安装且可用, 跳过"
} else {
    switch ($Device) {
        "CU128" { $idx = "https://download.pytorch.org/whl/cu128" }
        "CU126" { $idx = "https://download.pytorch.org/whl/cu126" }
        "CPU"   { $idx = "https://download.pytorch.org/whl/cpu" }
    }
    & uv pip install --python $Py torch torchaudio --index-url $idx
    if ($LASTEXITCODE -ne 0) { throw "PyTorch 安装失败" }
}
Success "PyTorch 就绪"

# ============================================================
# Step 5: 安装项目依赖 (opencc等需MSVC编译, 自动激活vcvars64)
# ============================================================
Step "Step 5/9  安装项目依赖 (extra-req + requirements)"
# 5a. 定位 VS 的 vcvars64.bat (供 opencc/jieba_fast/pyopenjtalk 源码编译)
$vcvars = $null
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($vsPath) { $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat" }
}
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    foreach ($ed in "Community", "Professional", "Enterprise", "BuildTools") {
        $cand = "C:\Program Files\Microsoft Visual Studio\2022\$ed\VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $cand) { $vcvars = $cand; break }
    }
}

& uv pip install --python $Py -r extra-req.txt --no-deps
if ($LASTEXITCODE -ne 0) { throw "extra-req.txt 安装失败" }

if ($vcvars) {
    Info "检测到 MSVC, 在 vcvars64 环境下编译安装 requirements.txt"
    # 通过临时bat激活vcvars后调用uv, 使opencc等能找到cl.exe
    $tmpBat = Join-Path $env:TMP "_gsv_reqs.bat"
    @"
@echo off
call "$vcvars"
set "UV_CACHE_DIR=$($env:UV_CACHE_DIR)"
set "UV_PYTHON_INSTALL_DIR=$($env:UV_PYTHON_INSTALL_DIR)"
set "TMP=$($env:TMP)"
set "TEMP=$($env:TEMP)"
set "PIP_CACHE_DIR=$($env:PIP_CACHE_DIR)"
cd /d "$ScriptDir"
uv pip install --python "$Py" -r requirements.txt
exit /b %errorlevel%
"@ | Set-Content -Path $tmpBat -Encoding Ascii
    & cmd /c "`"$tmpBat`""
    $reqCode = $LASTEXITCODE
    Remove-Item $tmpBat -Force -ErrorAction SilentlyContinue
    if ($reqCode -ne 0) { throw "requirements.txt 安装失败 (code=$reqCode)" }
} else {
    Warn "未找到 MSVC (vcvars64.bat), 尝试直接安装 (opencc若编译失败请装 VS2022 C++ 生成工具)"
    & uv pip install --python $Py -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "requirements.txt 安装失败" }
}
Success "项目依赖就绪"

# ============================================================
# Step 5c: 修正依赖漂移 (requirements.txt 未锁 fastapi 上限)
#   gradio 4.44.1 用旧式 TemplateResponse(name, context) 调用,
#   与最新 starlette(>=1.x, 仅支持 request 优先签名) 不兼容,
#   访问首页会报 TypeError: unhashable type: 'dict'.
#   钉住 fastapi 0.115.2 -> 连带 starlette 0.40.x (保留旧式兼容).
# ============================================================
Step "Step 5c  修正 fastapi/starlette 版本 (兼容 gradio 4.44.1)"
$fastapiOK = $false
$fv = (& $Py -c "import fastapi;print(fastapi.__version__)" 2>$null)
if ($LASTEXITCODE -eq 0 -and $fv -eq "0.115.2") { $fastapiOK = $true }
if ($fastapiOK) {
    Info "fastapi 已是 0.115.2, 跳过"
} else {
    & uv pip install --python $Py "fastapi[standard]==0.115.2"
    if ($LASTEXITCODE -ne 0) { throw "fastapi 版本修正失败" }
}
Success "fastapi/starlette 版本已修正"

# ============================================================
# Step 6: 配置 ffmpeg/ffprobe (复制到项目根)
# ============================================================
Step "Step 6/9  配置 ffmpeg / ffprobe"
foreach ($exe in "ffmpeg", "ffprobe") {
    $dst = Join-Path $ScriptDir "$exe.exe"
    if (Test-Path $dst) { Info "$exe.exe 已存在, 跳过"; continue }
    $cmd = Get-Command "$exe" -ErrorAction SilentlyContinue
    if ($cmd) { Copy-Item $cmd.Source $dst -Force; Success "已复制 $exe.exe" }
    else { Warn "PATH 中未找到 $exe, 请手动放置 $exe.exe 到项目根" }
}

# ============================================================
# Step 7: 下载模型资源 (断点续传, 已存在则跳过)
# ============================================================
Step "Step 7/9  下载模型资源 (源: $Source)"
function Get-File($name) {
    $out = Join-Path $env:TMP $name
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 0)) { Info "$name 已下载, 续传校验"; }
    & curl.exe -L -C - --retry 8 --retry-delay 5 -# -o $out "$Base/$name"
    if ($LASTEXITCODE -ne 0) { throw "$name 下载失败" }
    return $out
}
# 下载清单: 文件名 -> 解压目标(相对项目根) | "存在判据"目录
$models = @(
    @{ file = "pretrained_models.zip"; dest = "GPT_SoVITS";      check = "GPT_SoVITS\pretrained_models\sv" },
    @{ file = "G2PWModel.zip";         dest = "GPT_SoVITS\text"; check = "GPT_SoVITS\text\G2PWModel" },
    @{ file = "nltk_data.zip";         dest = $CacheRoot;        check = "$CacheRoot\nltk_data\corpora" }
)
if (-not $NoUVR5) {
    $models += @{ file = "uvr5_weights.zip"; dest = "tools\uvr5"; check = "tools\uvr5\uvr5_weights" }
}
$openjtalkDir = (& $Py -c "import os,pyopenjtalk;print(os.path.dirname(pyopenjtalk.__file__))").Trim()

$toExtract = @()
foreach ($m in $models) {
    $checkPath = if ([System.IO.Path]::IsPathRooted($m.check)) { $m.check } else { Join-Path $ScriptDir $m.check }
    if (Test-Path $checkPath) { Info "$($m.file) 目标已存在, 跳过下载"; continue }
    $zip = Get-File $m.file
    $toExtract += @{ zip = $zip; dest = $m.dest }
}
# open_jtalk 字典 (解压到 pyopenjtalk 包目录)
$ojCheck = Join-Path $openjtalkDir "open_jtalk_dic_utf_8-1.11"
if (-not (Test-Path $ojCheck)) {
    $ojZip = Get-File "open_jtalk_dic_utf_8-1.11.tar.gz"
    $toExtract += @{ zip = $ojZip; dest = $openjtalkDir; tar = $true }
} else { Info "open_jtalk 字典已存在, 跳过" }
Success "模型下载完成"

# ============================================================
# Step 8: 解压 (用Python的zipfile/tarfile, 避免tar的host:path陷阱)
# ============================================================
Step "Step 8/9  解压模型资源"
foreach ($e in $toExtract) {
    $isTar = [bool]$e.tar
    Info "解压 $([System.IO.Path]::GetFileName($e.zip)) -> $($e.dest)"
    $pycode = if ($isTar) {
        "import tarfile,os; os.makedirs(r'$($e.dest)',exist_ok=True); tarfile.open(r'$($e.zip)','r:gz').extractall(r'$($e.dest)')"
    } else {
        "import zipfile,os; os.makedirs(r'$($e.dest)',exist_ok=True); zipfile.ZipFile(r'$($e.zip)').extractall(r'$($e.dest)')"
    }
    & $Py -c $pycode
    if ($LASTEXITCODE -ne 0) { throw "解压失败: $($e.zip)" }
    Remove-Item $e.zip -Force -ErrorAction SilentlyContinue
}
Success "解压完成, 临时压缩包已清理"

# ============================================================
# Step 9: 生成启动脚本 + 验证
# ============================================================
Step "Step 9/9  生成 run-webui.bat 并验证"

# 9a. 生成启动脚本 (注入缓存变量 + 用.venv启动)
$runBat = Join-Path $ScriptDir "run-webui.bat"
@"
@echo off
chcp 65001 >nul
REM ====== GPT-SoVITS WebUI 启动脚本 (setup-local.ps1 自动生成) ======
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "CACHE_ROOT=$CacheRoot"
set "TMP=%CACHE_ROOT%\tmp"
set "TEMP=%CACHE_ROOT%\tmp"
set "HF_HOME=%CACHE_ROOT%\huggingface"
set "HF_ENDPOINT=$HFEndpoint"
set "MODELSCOPE_CACHE=%CACHE_ROOT%\modelscope"
set "TORCH_HOME=%CACHE_ROOT%\torch"
set "NLTK_DATA=%CACHE_ROOT%\nltk_data"
set "PATH=%SCRIPT_DIR%;%PATH%"
"%SCRIPT_DIR%\.venv\Scripts\python.exe" -I webui.py zh_CN
pause
"@ | Set-Content -Path $runBat -Encoding Ascii
Success "已生成 $runBat"

# 9b. 验证环境
Info "验证 torch CUDA 与关键依赖..."
& $Py -c @"
import torch, torchaudio, transformers, gradio, funasr, faster_whisper, onnxruntime
import opencc, jieba_fast, pyopenjtalk, librosa, ctranslate2
print('  torch       :', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.version.cuda)
print('  GPU         :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('  onnxruntime :', onnxruntime.get_available_providers())
print('  transformers:', transformers.__version__, '| gradio:', gradio.__version__)
pyopenjtalk.g2p('テスト')
print('  编译包/字典  : opencc/jieba_fast/pyopenjtalk/librosa OK')
"@
if ($LASTEXITCODE -ne 0) { throw "环境验证失败" }

Write-Host "`n============================================================" -ForegroundColor Green
Success "安装全部完成! 双击 run-webui.bat 即可启动 WebUI"
Success "所有缓存位于 $CacheRoot (C盘零写入)"
Write-Host "============================================================" -ForegroundColor Green



