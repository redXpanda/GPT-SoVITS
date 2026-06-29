# ============================================================
# GPT-SoVITS 统一环境变量脚本 (浮浮酱定制)
# 作用: 把所有模型缓存 / 临时文件 / 解释器全部劫持到 F:\ai\cache
#       严禁向 C 盘默认路径写入任何文件 (C盘仅剩极少空间)
# 用法: 安装期与运行期都先 . .\_env.ps1 (点号source) 注入变量
# ============================================================

$CacheRoot = "F:\ai\cache"

# ---- uv: 包缓存 + 下载的 Python 解释器 (默认均落 C 盘) ----
$env:UV_CACHE_DIR          = "$CacheRoot\uv"
$env:UV_PYTHON_INSTALL_DIR = "$CacheRoot\uv-python"

# ---- 临时目录: pip/uv 解压大包(torch ~2.5G)默认用 C 盘 %TEMP% ----
$env:TMP  = "$CacheRoot\tmp"
$env:TEMP = "$CacheRoot\tmp"

# ---- pip 缓存 (保险, 即便用 uv 也兜底) ----
$env:PIP_CACHE_DIR = "$CacheRoot\pip"

# ---- HuggingFace: BERT / faster-whisper / transformers 等 ----
$env:HF_HOME     = "$CacheRoot\huggingface"
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内镜像加速

# ---- ModelScope: funasr ASR / eres2net 等 ----
$env:MODELSCOPE_CACHE = "$CacheRoot\modelscope"

# ---- PyTorch Hub ----
$env:TORCH_HOME = "$CacheRoot\torch"

# ---- NLTK (g2p_en 依赖) ----
$env:NLTK_DATA = "$CacheRoot\nltk_data"

Write-Host "[ENV] 缓存全部重定向至 $CacheRoot (禁写C盘)" -ForegroundColor Green
