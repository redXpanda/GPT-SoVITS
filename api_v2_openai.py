"""
GPT-SoVITS OpenAI-Compatible TTS API
=====================================

在 GPT-SoVITS 之上提供 OpenAI 兼容的 ``/v1/audio/speech`` 接口，支持流式与完整
两种返回模式，音色库直接复用本项目 ``presets/<name>.json`` 体系。

设计要点(参考 VoxCPM/api.py，保持四层解耦，便于后续替换引擎):
- ``TTSEngine``  : 引擎无关协议，逐段产出 (sample_rate, int16 PCM)。
- ``LocalGPTSoVITSEngine`` : 包装本项目 ``tts_pipeline``；由于每个 voice 对应一套
  独立微调模型，引擎内部缓存"当前已加载 preset"，仅在 voice 变化时切换权重。
- ``VoiceLibrary`` : 把 ``presets/*.json`` 解析为引擎参数(参考音频/文本/语言)。
- ``AudioEncoder`` : int16 PCM -> mp3(ffmpeg) / wav / pcm，流式与完整通用。

运行:
    python api_v2_openai.py -a 0.0.0.0 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml

OpenAI 客户端示例:
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:9880/v1", api_key="sk-anything")
    resp = client.audio.speech.create(model="gpt-sovits", voice="okayu",
                                       input="こんにちは", response_format="mp3")
    resp.stream_to_file("out.mp3")
"""

import os
import sys
import json
import glob
import struct
import signal
import logging
import argparse
import subprocess
import traceback
from io import BytesIO
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional, Protocol, Union

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gptsovits.openai")

i18n = I18nAuto()


# --------------------------------------------------------------------------- #
# 启动参数与全局 TTS 管线初始化
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="GPT-SoVITS OpenAI-compatible api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=9880, help="default: 9880")
args = parser.parse_args()

config_path = args.tts_config
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


# --------------------------------------------------------------------------- #
# 配置(默认生成参数，preset / 请求未指定时的回退值)
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    model_name: str = "gpt-sovits"   # 对外暴露的模型名
    default_voice: str = ""          # 默认 voice(空=自动取首个 preset)
    text_lang: str = "auto"          # 输出语言(OpenAI 无此字段，用 auto 多语种识别)
    # 合成默认参数
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    speed_factor: float = 1.0


settings = Settings()


# --------------------------------------------------------------------------- #
# 引擎无关的内部请求(OpenAI 字段 -> 内部 TTSRequest 的解耦边界)
# --------------------------------------------------------------------------- #
@dataclass
class TTSRequest:
    """引擎无关的合成请求。OpenAI 适配层 + 音色库共同填充，引擎只消费它。"""

    text: str
    text_lang: str = "auto"
    ref_audio_path: Optional[str] = None
    prompt_text: str = ""
    prompt_lang: str = ""
    speed: float = 1.0
    preset_name: Optional[str] = None  # 用于引擎判断是否需要切换模型权重


# --------------------------------------------------------------------------- #
# OpenAI 兼容的 HTTP 请求体(字段严格对齐 OpenAI /v1/audio/speech)
# --------------------------------------------------------------------------- #
class SpeechRequest(BaseModel):
    model: str = Field(default="gpt-sovits", description="模型名(本服务固定一种，仅作兼容)")
    input: str = Field(..., max_length=4096, description="要合成的文本，最长 4096 字符")
    voice: str = Field(default="", description="音色名 -> 映射到 preset")
    response_format: str = Field(default="mp3", description="mp3 / wav / pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="语速")
    instructions: Optional[str] = Field(default=None, description="语气指令(GPT-SoVITS 暂未使用)")
    stream_format: Optional[str] = Field(default=None, description="audio=分块流式；None/sse 走完整")
    stream: Optional[bool] = Field(default=None, description="便捷开关：true 等价 stream_format=audio")



# --------------------------------------------------------------------------- #
# ① 引擎抽象层(引擎无关接口)
# --------------------------------------------------------------------------- #
class TTSEngine(Protocol):
    """引擎无关协议。逐段产出 (sample_rate, int16 PCM 1-D ndarray)。"""

    sample_rate: int

    def synthesize_stream(self, req: TTSRequest) -> Iterator[tuple]:
        ...


class LocalGPTSoVITSEngine:
    """包装本项目 ``tts_pipeline``。每个 voice 是独立微调模型，
    内部缓存"当前已加载 preset"，仅在切换 voice 时重载权重(耗时)。"""

    def __init__(self, pipeline: TTS):
        self._pipeline = pipeline
        self._loaded_preset: Optional[str] = None
        self.sample_rate: int = 32000  # 实际 sr 以每次产出为准，这里仅作 header 默认

    def ensure_preset(self, preset_name: str, gpt_weights: str, sovits_weights: str):
        """确保指定 preset 的模型已加载；已是当前 preset 则跳过(避免重复重载)喵～"""
        if preset_name and preset_name == self._loaded_preset:
            return
        if gpt_weights:
            logger.info("Loading GPT weights for preset '%s': %s", preset_name, gpt_weights)
            self._pipeline.init_t2s_weights(gpt_weights)
        if sovits_weights:
            logger.info("Loading SoVITS weights for preset '%s': %s", preset_name, sovits_weights)
            self._pipeline.init_vits_weights(sovits_weights)
        self._loaded_preset = preset_name

    @property
    def loaded_preset(self) -> Optional[str]:
        return self._loaded_preset

    def synthesize_stream(self, req: TTSRequest) -> Iterator[tuple]:
        run_req = {
            "text": req.text,
            "text_lang": (req.text_lang or "auto").lower(),
            "ref_audio_path": req.ref_audio_path,
            "aux_ref_audio_paths": None,
            "prompt_text": req.prompt_text,
            "prompt_lang": (req.prompt_lang or "").lower(),
            "top_k": settings.top_k,
            "top_p": settings.top_p,
            "temperature": settings.temperature,
            "text_split_method": settings.text_split_method,
            "batch_size": settings.batch_size,
            "speed_factor": float(req.speed or settings.speed_factor),
            "split_bucket": True,
            "return_fragment": True,  # 开启分段产出以支持流式
            "fragment_interval": 0.3,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": 32,
            "super_sampling": False,
        }
        for sr, chunk in self._pipeline.run(run_req):
            self.sample_rate = sr
            arr = np.asarray(chunk).reshape(-1)
            if arr.size:
                yield sr, arr



# --------------------------------------------------------------------------- #
# ② 音色库(复用 presets/*.json + slicer.list 解析)
# --------------------------------------------------------------------------- #
@dataclass
class VoiceInfo:
    name: str
    display_name: str
    has_reference: bool
    prompt_lang: str = ""
    description: str = ""


def parse_ref_list(list_path: str):
    """解析 GPT-SoVITS 标注 list 文件: 路径|说话人|语言(大写)|文本，过滤空文本喵～"""
    items = []
    if not list_path or not os.path.exists(list_path):
        return items
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            audio_path, _spk, lang, text = parts[0], parts[1], parts[2], "|".join(parts[3:])
            text = text.strip()
            if not text:
                continue
            items.append({"audio_path": audio_path, "lang": lang.strip().lower(), "text": text})
    return items


class VoiceLibrary:
    """把 presets/<name>.json 解析为引擎参数。voice 名直接对应 preset 名。"""

    def _preset_names(self) -> List[str]:
        os.makedirs(PRESETS_DIR, exist_ok=True)
        return [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(PRESETS_DIR, "*.json")))]

    def _load_json(self, name: str) -> Optional[dict]:
        fpath = os.path.join(PRESETS_DIR, os.path.basename(name) + ".json")
        if not os.path.exists(fpath):
            return None
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("preset '%s' parse failed: %s", name, e)
            return None

    def list_voices(self) -> List[VoiceInfo]:
        voices: List[VoiceInfo] = []
        for name in self._preset_names():
            data = self._load_json(name) or {}
            ref = self._pick_ref(data)
            voices.append(
                VoiceInfo(
                    name=name,
                    display_name=data.get("display_name", name),
                    has_reference=bool(ref and ref.get("audio_path") and os.path.exists(ref["audio_path"])),
                    prompt_lang=data.get("prompt_lang", ""),
                    description=(ref.get("text", "") if ref else "")[:60],
                )
            )
        return voices

    def default_voice(self) -> Optional[str]:
        if settings.default_voice:
            return settings.default_voice
        names = self._preset_names()
        return names[0] if names else None

    def _pick_ref(self, data: dict) -> Optional[dict]:
        """挑选参考音频条目: 优先 default_ref_audio，否则 list 第一条；返回 {audio_path,text,lang}。"""
        items = parse_ref_list(data.get("ref_list_file", ""))
        default_path = data.get("default_ref_audio", "")
        if default_path:
            for it in items:
                if os.path.abspath(it["audio_path"]) == os.path.abspath(default_path):
                    return it
            # default_ref_audio 不在 list 中也允许直接使用(文本回退 preset 无则空)
            return {"audio_path": default_path, "text": "", "lang": data.get("prompt_lang", "")}
        return items[0] if items else None

    def resolve(self, voice: str) -> Optional[dict]:
        """把 voice 名解析为模型路径 + 参考音频参数；返回 None 表示未知音色。"""
        name = voice or self.default_voice()
        if not name:
            return None
        data = self._load_json(name)
        if data is None:
            # 未知 voice -> 回退默认 preset
            name = self.default_voice()
            data = self._load_json(name) if name else None
            if data is None:
                return None
        ref = self._pick_ref(data)
        if ref is None:
            return None
        return {
            "preset_name": name,
            "gpt_weights": data.get("gpt_weights", ""),
            "sovits_weights": data.get("sovits_weights", ""),
            "ref_audio_path": ref["audio_path"],
            "prompt_text": ref.get("text", ""),
            "prompt_lang": ref.get("lang", "") or data.get("prompt_lang", ""),
        }



# --------------------------------------------------------------------------- #
# ③ 音频编码层(int16 PCM -> mp3 / wav / pcm；流式与完整通用)
# --------------------------------------------------------------------------- #
_FORMAT_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pcm": "audio/L16",  # 裸 16-bit PCM
}


def _to_s16_bytes(pcm: np.ndarray) -> bytes:
    """int16(或其它) ndarray -> 16-bit little-endian PCM 字节。"""
    arr = np.asarray(pcm)
    if arr.dtype != np.int16:
        arr = np.clip(arr, -32768, 32767).astype("<i2")
    else:
        arr = arr.astype("<i2")
    return arr.tobytes()


def _wav_header(sample_rate: int, channels: int = 1, bits: int = 16, data_size: int = 0xFFFFFFFF) -> bytes:
    """构造 WAV 头。流式时 data_size 用占位最大值(多数播放器容忍)。"""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    riff_size = 36 + data_size if data_size != 0xFFFFFFFF else 0xFFFFFFFF
    return (
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", data_size)
    )


class AudioEncoder:
    """把引擎产出的 int16 PCM chunk 编码为目标容器格式。"""

    @staticmethod
    def normalize_format(fmt: str) -> str:
        fmt = (fmt or "mp3").lower()
        # 不支持的 OpenAI 格式(opus/aac/flac)回退到 mp3，保证兼容
        return fmt if fmt in _FORMAT_MIME else "mp3"

    def mime(self, fmt: str) -> str:
        return _FORMAT_MIME[self.normalize_format(fmt)]

    # ---- 流式编码：逐 chunk 产出字节 ---- #
    def stream(self, chunks: Iterator[tuple], fmt: str) -> Iterator[bytes]:
        fmt = self.normalize_format(fmt)
        if fmt == "mp3":
            yield from self._stream_mp3(chunks)
        elif fmt == "wav":
            first = True
            for sr, c in chunks:
                if first:
                    yield _wav_header(sr, data_size=0xFFFFFFFF)
                    first = False
                yield _to_s16_bytes(c)
        else:  # pcm
            for _sr, c in chunks:
                yield _to_s16_bytes(c)

    def _stream_mp3(self, chunks: Iterator[tuple]) -> Iterator[bytes]:
        """用 ffmpeg subprocess 把 s16le PCM 流编码为 mp3(对齐项目内 pack_aac 模式)喵～"""
        proc = None
        sr_holder = {"sr": 32000}
        # 先取第一块拿到采样率再起 ffmpeg
        it = iter(chunks)
        try:
            first = next(it)
        except StopIteration:
            return
        sr_holder["sr"] = first[0]
        proc = subprocess.Popen(
            [
                "ffmpeg", "-f", "s16le", "-ar", str(sr_holder["sr"]), "-ac", "1",
                "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "pipe:1",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        # 由于 ffmpeg 管道是阻塞的，这里采用"全部写入->读取"策略(简洁稳妥)
        proc.stdin.write(_to_s16_bytes(first[1]))
        for _sr, c in it:
            proc.stdin.write(_to_s16_bytes(c))
        out, _ = proc.communicate()
        if out:
            yield out

    # ---- 完整编码：拼接后一次性返回 ---- #
    def encode_full(self, chunks: Iterator[tuple], fmt: str) -> bytes:
        fmt = self.normalize_format(fmt)
        sr = 32000
        parts = []
        for s, c in chunks:
            sr = s
            parts.append(np.asarray(c).reshape(-1))
        pcm = np.concatenate(parts) if parts else np.zeros(1, dtype=np.int16)
        body = _to_s16_bytes(pcm)
        if fmt == "wav":
            buf = BytesIO()
            sf.write(buf, np.frombuffer(body, dtype="<i2"), sr, format="wav")
            return buf.getvalue()
        elif fmt == "mp3":
            proc = subprocess.Popen(
                [
                    "ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1",
                    "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "pipe:1",
                ],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            out, _ = proc.communicate(input=body)
            return out
        else:  # pcm
            return body



# --------------------------------------------------------------------------- #
# ④ FastAPI 应用与路由(OpenAI 兼容)
# --------------------------------------------------------------------------- #
_voices = VoiceLibrary()
_engine: TTSEngine = LocalGPTSoVITSEngine(tts_pipeline)
_encoder = AudioEncoder()

APP = FastAPI(title="GPT-SoVITS OpenAI-Compatible TTS API", version="1.0")

# 允许跨域访问，方便前端/网页直接调用本 API 喵～
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_request(body: SpeechRequest) -> TTSRequest:
    """OpenAI 请求体 + 音色库解析结果 -> 引擎无关 TTSRequest，并确保模型已加载。"""
    resolved = _voices.resolve(body.voice)
    if resolved is None:
        raise HTTPException(status_code=400, detail=f"voice '{body.voice}' not found and no preset available")
    # 切换/确保 preset 模型已加载(仅 voice 变化时重载)
    _engine.ensure_preset(resolved["preset_name"], resolved["gpt_weights"], resolved["sovits_weights"])
    return TTSRequest(
        text=body.input,
        text_lang=settings.text_lang,
        ref_audio_path=resolved["ref_audio_path"],
        prompt_text=resolved["prompt_text"],
        prompt_lang=resolved["prompt_lang"],
        speed=body.speed,
        preset_name=resolved["preset_name"],
    )


def _is_streaming(body: SpeechRequest) -> bool:
    if body.stream is not None:
        return bool(body.stream)
    return (body.stream_format or "").lower() == "audio"


@APP.post("/v1/audio/speech")
def create_speech(body: SpeechRequest):
    if not body.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty")

    req = _build_request(body)
    fmt = _encoder.normalize_format(body.response_format)
    mime = _encoder.mime(fmt)
    headers = {"X-Audio-Channels": "1"}

    if _is_streaming(body):
        logger.info("/v1/audio/speech streaming voice=%s fmt=%s", req.preset_name, fmt)

        def gen() -> Iterator[bytes]:
            try:
                yield from _encoder.stream(_engine.synthesize_stream(req), fmt)
            except Exception as e:
                logger.exception("streaming synthesis failed: %s", e)

        return StreamingResponse(gen(), media_type=mime, headers=headers)

    logger.info("/v1/audio/speech full voice=%s fmt=%s", req.preset_name, fmt)
    try:
        audio_bytes = _encoder.encode_full(_engine.synthesize_stream(req), fmt)
    except Exception as e:
        logger.exception("synthesis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")
    headers["Content-Length"] = str(len(audio_bytes))
    return Response(content=audio_bytes, media_type=mime, headers=headers)


@APP.get("/v1/audio/voices")
def list_voices():
    items = [
        {
            "id": v.name,
            "name": v.name,
            "display_name": v.display_name,
            "has_reference": v.has_reference,
            "prompt_lang": v.prompt_lang,
            "description": v.description,
        }
        for v in _voices.list_voices()
    ]
    return {"object": "list", "data": items}


@APP.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": settings.model_name, "object": "model", "owned_by": "gpt-sovits"}],
    }


@APP.get("/health")
def health():
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "loaded_preset": _engine.loaded_preset, "sample_rate": _engine.sample_rate},
    )


@APP.get("/", response_class=HTMLResponse)
def test_page():
    """测试台页面(同源，零 CORS)。"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_v2_openai.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>api_v2_openai.html not found</h1>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    try:
        if host == "None":  # -a None 可监听双栈
            _host = None
        else:
            _host = host
        uvicorn.run(app=APP, host=_host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)





