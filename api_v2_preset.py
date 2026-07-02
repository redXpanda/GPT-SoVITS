"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 15,                  # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
    "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
    "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
    "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
    "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

"""

import os
import sys
import json
import glob
import traceback
from typing import Generator, Union

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
import threading

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
parser.add_argument(
    "-pre",
    "--preset",
    type=str,
    default="",
    help="启动时自动加载的默认 preset 名称(presets/ 下的文件名, 可不含 .json), 留空则不加载",
)
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()

# 允许跨域访问，方便前端/网页直接调用本 API 喵～
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    # Author: AkagawaTsurunaki
    # Issue:
    #   Stack overflow probabilistically occurs
    #   when the function `sf_writef_short` of `libsndfile_64bit.dll` is called
    #   using the Python library `soundfile`
    # Note:
    #   This is an issue related to `libsndfile`, not this project itself.
    #   It happens when you generate a large audio tensor (about 499804 frames in my PC)
    #   and try to convert it to an ogg file.
    # Related:
    #   https://github.com/RVC-Boss/GPT-SoVITS/issues/1199
    #   https://github.com/libsndfile/libsndfile/issues/1023
    #   https://github.com/bastibe/python-soundfile/issues/396
    # Suggestion:
    #   Or split the whole audio data into smaller audio segment to avoid stack overflow?

    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)



    # See: https://docs.python.org/3/library/threading.html
    # The stack size of this thread is at least 32768
    # If stack overflow error still occurs, just modify the `stack_size`.
    # stack_size = n * 4096, where n should be a positive integer.
    # Here we chose n = 4096.
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")

    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    # elif media_type == "ogg" and not streaming_mode:
    #     return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 15,                  # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "parallel_infer": True,       # bool. whether to use parallel inference.
                "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
                "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
                "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
                "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
                "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    # 未传参考音频时, 回退到当前已加载 preset 的默认参考音频/文本/语言喵～
    # 以「参考音频为空」为触发: 没音频则整套用默认; 文本/语言仅在同样为空时补默认, 尊重显式传入的值
    if not req.get("ref_audio_path") and CURRENT_PRESET.get("default_ref_audio"):
        req["ref_audio_path"] = CURRENT_PRESET["default_ref_audio"]
        if not req.get("prompt_text"):
            req["prompt_text"] = CURRENT_PRESET["default_ref_text"]
        if not req.get("prompt_lang"):
            req["prompt_lang"] = CURRENT_PRESET["default_ref_lang"]
    # 参考音频路径已指定但文件不存在时，回退到默认参考音频并打警告日志喵～
    elif req.get("ref_audio_path") and not os.path.exists(req["ref_audio_path"]):
        missing_path = req["ref_audio_path"]
        if CURRENT_PRESET.get("default_ref_audio"):
            print(
                f"[WARNING] 参考音频不存在: {missing_path}，"
                f"已回退到默认参考音频: {CURRENT_PRESET['default_ref_audio']}"
            )
            req["ref_audio_path"] = CURRENT_PRESET["default_ref_audio"]
            # 原音频不存在，随之传来的 prompt_text/lang 已无效，无条件替换为默认值喵～
            req["prompt_text"] = CURRENT_PRESET["default_ref_text"]
            req["prompt_lang"] = CURRENT_PRESET["default_ref_lang"]
        else:
            print(f"[WARNING] 参考音频不存在: {missing_path}，且当前无默认参考音频可回退喵～")

    check_res = check_params(req)
    if check_res is not None:
        return check_res
    
    if streaming_mode == 0:
        streaming_mode = False
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 1:
        streaming_mode = False
        return_fragment = True
        fixed_length_chunk = False
    elif streaming_mode == 2:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 3:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = True

    else:
        return JSONResponse(status_code=400, content={"message": f"the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)"})

    req["streaming_mode"] = streaming_mode
    req["return_fragment"] = return_fragment
    req["fixed_length_chunk"] = fixed_length_chunk

    print(f"{streaming_mode} {return_fragment} {fixed_length_chunk}")

    streaming_mode = streaming_mode or return_fragment


    try:
        tts_generator = tts_pipeline.run(req)

        if streaming_mode:

            def streaming_generator(tts_generator: Generator, media_type: str):
                if_frist_chunk = True
                for sr, chunk in tts_generator:
                    if if_frist_chunk and media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        media_type = "raw"
                        if_frist_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(
                streaming_generator(
                    tts_generator,
                    media_type,
                ),
                media_type=f"audio/{media_type}",
            )

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
    streaming_mode: Union[bool, int] = False,
    overlap_length: int = 2,
    min_chunk_length: int = 16,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
        "overlap_length": int(overlap_length),
        "min_chunk_length": int(min_chunk_length),
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# ============================================================================
# 以下为内置 HTML 测试页面相关接口 (原 api_v2_official.py 中没有，本文件新增) 喵～
# ============================================================================


@APP.get("/tts_config")
async def get_tts_config():
    """返回当前 TTS 配置信息，供内置测试页面动态填充表单喵～"""
    return JSONResponse(
        status_code=200,
        content={
            "version": str(tts_config.version),
            "languages": list(tts_config.languages),
            "cut_methods": list(cut_method_names),
            "device": str(tts_config.device),
            "is_half": bool(tts_config.is_half),
        },
    )


@APP.get("/", response_class=HTMLResponse)
async def index():
    """内置 API 测试台页面，读取同目录下的 api_v2_preset.html 返回喵～"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_v2_preset.html")
    if not os.path.exists(html_path):
        return HTMLResponse(
            content="<h1>测试页面缺失</h1><p>未找到 api_v2_preset.html，请确认它与本脚本在同一目录喵～</p>",
            status_code=404,
        )
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ============================================================================
# 以下为角色预设(preset)相关接口 喵～
# preset 是 presets/ 目录下的 json 文件，记录模型路径与参考音频 list 文件路径，
# 一键加载即可切换模型并获得可选的参考音频列表，省去手填一堆配置。
# ============================================================================

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

# 试听白名单: 仅允许播放已被 preset 解析过的参考音频，防止任意文件读取喵～
ALLOWED_REF_AUDIOS = set()

# 当前已加载 preset 的状态(供 /speakers 返回其 slice 语音列表)喵～
# default_ref_audio / default_ref_text / default_ref_lang 供 /tts_to_audio 覆盖参考音频时使用
CURRENT_PRESET = {
    "name": None,
    "ref_audios": [],
    "default_ref_audio": "",
    "default_ref_text": "",
    "default_ref_lang": "",
}


def parse_ref_list(list_path: str):
    """解析 GPT-SoVITS 标注 list 文件: 路径|说话人|语言(大写)|文本 ，返回可用参考音频列表喵～"""
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
            if not text:  # 参考音频必须带文本，跳过空文本条目喵～
                continue
            exists = os.path.exists(audio_path)
            if exists:
                ALLOWED_REF_AUDIOS.add(os.path.abspath(audio_path))
            items.append(
                {
                    "audio_path": audio_path,
                    "lang": lang.strip().lower(),
                    "text": text,
                    "exists": exists,
                }
            )
    return items


@APP.get("/ref_audio")
async def get_ref_audio(path: str = None):
    """返回参考音频文件用于试听，仅限已被 preset 解析过的白名单路径喵～"""
    if path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "path is required"})
    abspath = os.path.abspath(path)
    if abspath not in ALLOWED_REF_AUDIOS:
        return JSONResponse(status_code=403, content={"message": "audio not in allowed list, load a preset first"})
    if not os.path.exists(abspath):
        return JSONResponse(status_code=404, content={"message": "audio file not found"})
    return FileResponse(abspath)


def load_preset_file(name: str):
    """读取 presets/<name>.json，返回 (data, error_response)。"""
    safe_name = os.path.basename(name)  # 防止路径穿越喵～
    if not safe_name.endswith(".json"):
        safe_name += ".json"
    fpath = os.path.join(PRESETS_DIR, safe_name)
    if not os.path.exists(fpath):
        return None, JSONResponse(status_code=404, content={"message": f"preset not found: {safe_name}"})
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, JSONResponse(status_code=400, content={"message": "preset parse failed", "Exception": str(e)})


@APP.get("/presets")
async def list_presets():
    """列出 presets/ 目录下的所有预设(仅名称与基本信息)喵～"""
    os.makedirs(PRESETS_DIR, exist_ok=True)
    result = []
    for fpath in sorted(glob.glob(os.path.join(PRESETS_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(fpath))[0]
        info = {"name": name}
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            info["display_name"] = data.get("display_name", name)
            info["prompt_lang"] = data.get("prompt_lang", "")
        except Exception as e:
            info["error"] = str(e)
        result.append(info)
    return JSONResponse(status_code=200, content={"presets": result})


@APP.get("/preset")
async def get_preset(name: str = None):
    """返回指定预设详情 + 解析后的参考音频列表(不切换模型)喵～"""
    if name in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "name is required"})
    data, err = load_preset_file(name)
    if err is not None:
        return err
    ref_items = parse_ref_list(data.get("ref_list_file", ""))
    return JSONResponse(
        status_code=200,
        content={
            "name": name,
            "display_name": data.get("display_name", name),
            "gpt_weights": data.get("gpt_weights", ""),
            "sovits_weights": data.get("sovits_weights", ""),
            "prompt_lang": data.get("prompt_lang", ""),
            "ref_list_file": data.get("ref_list_file", ""),
            "ref_audios": ref_items,
        },
    )


def apply_preset(name: str):
    """加载预设核心逻辑(同步): 切换 GPT/SoVITS 模型并解析参考音频列表。

    返回 (result_dict, error_response)。供 /load_preset 端点与启动时自动加载复用喵～
    """
    data, err = load_preset_file(name)
    if err is not None:
        return None, err

    gpt_weights = data.get("gpt_weights", "")
    sovits_weights = data.get("sovits_weights", "")
    loaded = {"gpt": False, "sovits": False}
    try:
        if gpt_weights:
            tts_pipeline.init_t2s_weights(gpt_weights)
            loaded["gpt"] = True
        if sovits_weights:
            tts_pipeline.init_vits_weights(sovits_weights)
            loaded["sovits"] = True
    except Exception as e:
        return None, JSONResponse(
            status_code=400,
            content={"message": "load preset weights failed", "Exception": str(e), "loaded": loaded},
        )

    ref_items = parse_ref_list(data.get("ref_list_file", ""))
    # 记录当前已加载 preset，供 /speakers 返回其 slice 语音列表喵～
    CURRENT_PRESET["name"] = name
    CURRENT_PRESET["ref_audios"] = ref_items

    # 记录预设默认参考音频, 供 /tts_to_audio 在开关打开时覆盖请求参数喵～
    # 文本/语言优先从 ref_list 里匹配同一条音频, 匹配不到则文本留空、语言回退到 prompt_lang
    default_ref_audio = data.get("default_ref_audio", "")
    default_ref_text = ""
    default_ref_lang = data.get("prompt_lang", "")
    if default_ref_audio:
        ALLOWED_REF_AUDIOS.add(os.path.abspath(default_ref_audio))  # 加入白名单, 允许试听喵～
        for it in ref_items:
            if os.path.abspath(it["audio_path"]) == os.path.abspath(default_ref_audio):
                default_ref_text = it["text"]
                default_ref_lang = it["lang"]
                break
    CURRENT_PRESET["default_ref_audio"] = default_ref_audio
    CURRENT_PRESET["default_ref_text"] = default_ref_text
    CURRENT_PRESET["default_ref_lang"] = default_ref_lang
    result = {
        "message": "success",
        "name": name,
        "display_name": data.get("display_name", name),
        "prompt_lang": data.get("prompt_lang", ""),
        "loaded": loaded,
        "gpt_weights": gpt_weights,
        "sovits_weights": sovits_weights,
        "ref_audios": ref_items,
    }
    return result, None


@APP.get("/load_preset")
async def load_preset(name: str = None):
    """一键加载预设: 切换 GPT/SoVITS 模型，并返回参考音频列表喵～"""
    if name in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "name is required"})
    result, err = apply_preset(name)
    if err is not None:
        return err
    return JSONResponse(status_code=200, content=result)


@APP.get("/set_preset_default_ref")
async def set_preset_default_ref(name: str = None, audio_path: str = None):
    """将选中的参考音频设为 preset 的 default_ref_audio 并写回 json 文件喵～

    同时同步更新内存中 CURRENT_PRESET 状态（若该 preset 当前已加载）。
    """
    if name in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "name is required"})
    if audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "audio_path is required"})

    data, err = load_preset_file(name)
    if err is not None:
        return err

    data["default_ref_audio"] = audio_path

    safe_name = os.path.basename(name)
    if not safe_name.endswith(".json"):
        safe_name += ".json"
    fpath = os.path.join(PRESETS_DIR, safe_name)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "write preset failed", "Exception": str(e)})

    # 若该 preset 当前已加载，同步更新内存状态喵～
    if CURRENT_PRESET.get("name") == name:
        CURRENT_PRESET["default_ref_audio"] = audio_path
        ALLOWED_REF_AUDIOS.add(os.path.abspath(audio_path))
        # 从已解析的 ref_audios 里补 text/lang
        default_ref_text = ""
        default_ref_lang = data.get("prompt_lang", "")
        for it in CURRENT_PRESET["ref_audios"]:
            if os.path.abspath(it["audio_path"]) == os.path.abspath(audio_path):
                default_ref_text = it["text"]
                default_ref_lang = it["lang"]
                break
        CURRENT_PRESET["default_ref_text"] = default_ref_text
        CURRENT_PRESET["default_ref_lang"] = default_ref_lang

    return JSONResponse(status_code=200, content={"message": "success", "default_ref_audio": audio_path})


# ============================================================================
# 以下为从 api_v2.py 移植的额外接口，已适配本 preset 体系
# ============================================================================


def get_preset_names():
    """返回 presets/ 目录下所有预设名(不含 .json)，作为说话人来源喵～"""
    os.makedirs(PRESETS_DIR, exist_ok=True)
    return [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(PRESETS_DIR, "*.json")))]


# /tts_to_audio 是否用「当前已加载 preset 的默认参考音频」覆盖请求里的参考音频参数喵～
# False: 不覆盖, 完全尊重请求传入的 ref_audio_path/prompt_text/prompt_lang (当前默认)
# True : 用 CURRENT_PRESET 的 default_ref_audio/text/lang 覆盖, 调用方无需关心参考音频
TTS_TO_AUDIO_OVERRIDE_REF = False


@APP.post("/tts_to_audio/")
async def tts_to_audio(request: TTS_Request):
    # 原 api_v2.py 中此端点靠 config.llama_* 强制覆盖参考音频；本 preset 版去除 config 依赖，
    # 作为 POST /tts 的别名直接合成，仅保留原版 batch_size=10 的行为特征喵～
    req = request.dict()
    req["batch_size"] = 10

    # 开关打开且已加载 preset 时, 用预设默认参考音频覆盖请求参数喵～
    if TTS_TO_AUDIO_OVERRIDE_REF and CURRENT_PRESET.get("default_ref_audio"):
        req["ref_audio_path"] = CURRENT_PRESET["default_ref_audio"]
        req["prompt_text"] = CURRENT_PRESET["default_ref_text"]
        req["prompt_lang"] = CURRENT_PRESET["default_ref_lang"]

    return await tts_handle(req)


@APP.get("/speakers")
async def speakers_endpoint():
    # 数据源改为「当前已加载 preset 的 slice 语音列表」(原 api_v2.py 读 "参考音频" 目录)，
    # 返回结构与原版严格一致: [{"name": X, "voice_id": X}]，name 沿用原版去扩展名处理喵～
    voices = []
    for it in CURRENT_PRESET["ref_audios"]:
        name = os.path.basename(it["audio_path"])
        name = name.replace(".wav", "").replace(".mp3", "").replace(".WAV", "")
        voices.append({"name": name, "voice_id": name})
    return JSONResponse(voices, status_code=200)


@APP.get("/speakers_list")
def speakerlist_endpoint():
    # 基于 presets/ 列表返回名称数组，沿用 api_v2.py 的返回结构喵～
    return JSONResponse(get_preset_names(), status_code=200)


@APP.post("/")
async def tts_root_post_endpoint(request: TTS_Request):
    # 与 GET /(返回测试页面) HTTP 方法不同，可共存；行为等同 POST /tts 喵～
    req = request.dict()
    req["text_split_method"]='cut0'
    return await tts_handle(req)


if __name__ == "__main__":
    # 启动前若指定了默认 preset, 先加载模型再开启服务喵～
    if args.preset:
        print(f"[preset] 正在加载默认预设: {args.preset}")
        result, err = apply_preset(args.preset)
        if err is not None:
            # err 是 JSONResponse, 取出其 body 打印, 加载失败不阻断服务启动喵～
            print(f"[preset] 默认预设加载失败: {err.body.decode('utf-8', errors='ignore')}")
        else:
            print(
                f"[preset] 默认预设加载成功: {result['display_name']} "
                f"(GPT={result['loaded']['gpt']}, SoVITS={result['loaded']['sovits']}, "
                f"参考音频 {len(result['ref_audios'])} 条)"
            )

    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)