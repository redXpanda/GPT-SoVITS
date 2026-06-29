#requires -Version 7
<#
  修复 torchcodec 找不到 ffmpeg 共享库的问题喵～
  原理：torchcodec 的 libtorchcodec_core8.dll 依赖 ffmpeg 8 的 shared DLL，
        系统装的是 static 版（无 DLL），故下载 shared 版并把 DLL 拷进 torchcodec 目录。
  用 BtbN 的 win64-gpl-shared zip 包，PowerShell 原生解压，无需 7-Zip。
#>
$ErrorActionPreference = 'Stop'

$CacheDir      = 'F:/ai/cache/ffmpeg'
$Venv          = 'F:/ai/GPT-SOVITS-MASTER/GPT-SoVITS/.venv'
$TorchcodecDir = Join-Path $Venv 'Lib/site-packages/torchcodec'
# BtbN 的 ffmpeg 8.0 shared 构建（含 avcodec-62/avutil-60 等 DLL）
$Url     = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip'
$Archive = Join-Path $CacheDir 'ffmpeg-8.0-shared.zip'
$ExtractDir = Join-Path $CacheDir 'extracted'

New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

if (-not (Test-Path $Archive)) {
    Write-Host "下载 ffmpeg 8.0 shared build ..." -ForegroundColor Cyan
    # 用 curl 更快更稳；失败再退回 Invoke-WebRequest
    try { curl.exe -L -o $Archive $Url } catch { Invoke-WebRequest -Uri $Url -OutFile $Archive }
} else {
    Write-Host "已存在压缩包，跳过下载：$Archive" -ForegroundColor Yellow
}

Write-Host "解压中 ..." -ForegroundColor Cyan
if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
Expand-Archive -Path $Archive -DestinationPath $ExtractDir -Force

# 找到解压出的 bin 目录里的 DLL
$BinDir = Get-ChildItem -Path $ExtractDir -Recurse -Filter 'avcodec-*.dll' |
          Select-Object -First 1 | ForEach-Object { $_.DirectoryName }
if (-not $BinDir) { throw "没找到 avcodec dll，解压可能失败喵>_<" }

Write-Host "拷贝共享 DLL 到 torchcodec 目录：$TorchcodecDir" -ForegroundColor Cyan
Copy-Item -Path (Join-Path $BinDir '*.dll') -Destination $TorchcodecDir -Force

Write-Host "完成！现在验证喵～" -ForegroundColor Green
& "$Venv/Scripts/python.exe" -c @"
import torchaudio, numpy as np, soundfile as sf, tempfile, os
p = os.path.join(tempfile.gettempdir(), 'tc_test.wav')
sf.write(p, np.random.randn(16000).astype('float32'), 16000)
wav, sr = torchaudio.load(p)
print('OK load:', tuple(wav.shape), sr)
os.remove(p)
"@
