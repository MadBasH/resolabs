"""XTTS v2 TTS API - Multilingual TTS with voice cloning on Modal."""

import modal

R2_BUCKET_NAME = "resolabs-app"
R2_ACCOUNT_ID = "de31186ed80914fd17e2219c23eefa03"
R2_MOUNT_PATH = "/r2"
r2_bucket = modal.CloudBucketMount(
    R2_BUCKET_NAME,
    bucket_endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    secret=modal.Secret.from_name("cloudflare-r2"),
    read_only=True,
)

# Install in layers: torch first, then transformers pinned, then TTS
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.4.1", "torchaudio==2.4.1", "numpy<2.0")
    .pip_install("transformers==4.37.2")  # last version with BeamSearchScorer
    .pip_install("TTS==0.22.0", "fastapi[standard]==0.115.0")
)
app = modal.App("xtts-tts", image=image)

with image.imports():
    import io
    import os
    from pathlib import Path

    import torch
    import torchaudio as ta
    from TTS.api import TTS
    from fastapi import Depends, FastAPI, HTTPException, Security
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field

    SUPPORTED_LANGUAGES = [
        "tr", "en", "es", "fr", "de", "it", "pt", "pl",
        "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi", "ru"
    ]

    api_key_scheme = APIKeyHeader(name="x-api-key", scheme_name="ApiKeyAuth", auto_error=False)

    def verify_api_key(x_api_key: str | None = Security(api_key_scheme)):
        expected = os.environ.get("XTTS_API_KEY", "")
        if not expected or x_api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return x_api_key

    class TTSRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=5000)
        voice_key: str = Field(..., min_length=1, max_length=300)
        language: str = Field(default="tr")
        temperature: float = Field(default=0.75, ge=0.0, le=1.0)
        length_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
        repetition_penalty: float = Field(default=5.0, ge=1.0, le=10.0)
        top_k: int = Field(default=50, ge=1, le=100)
        top_p: float = Field(default=0.85, ge=0.0, le=1.0)
        speed: float = Field(default=1.0, ge=0.5, le=2.0)


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,
    secrets=[
        modal.Secret.from_name("xtts-api-key"),
        modal.Secret.from_name("cloudflare-r2"),
    ],
    volumes={R2_MOUNT_PATH: r2_bucket},
    timeout=120,
)
@modal.concurrent(max_inputs=5)
class XTTS:
    @modal.enter()
    def load_model(self):
        print("Loading XTTS v2 model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["COQUI_TOS_AGREED"] = "1"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print(f"XTTS v2 loaded on {self.device}")

    @modal.asgi_app()
    def serve(self):
        web_app = FastAPI(
            title="XTTS v2 TTS API",
            docs_url="/docs",
            dependencies=[Depends(verify_api_key)],
        )
        web_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

        @web_app.get("/languages")
        def list_languages():
            return {"supported_languages": SUPPORTED_LANGUAGES}

        @web_app.post("/generate", responses={200: {"content": {"audio/wav": {}}}})
        def generate_speech(request: TTSRequest):
            if request.language not in SUPPORTED_LANGUAGES:
                raise HTTPException(status_code=400, detail=f"Language '{request.language}' not supported.")
            voice_path = Path(R2_MOUNT_PATH) / request.voice_key
            if not voice_path.exists():
                raise HTTPException(status_code=400, detail=f"Voice file not found at '{request.voice_key}'")
            try:
                audio_bytes = self.generate.local(
                    text=request.prompt, speaker_wav=str(voice_path), language=request.language,
                    temperature=request.temperature, length_penalty=request.length_penalty,
                    repetition_penalty=request.repetition_penalty, top_k=request.top_k,
                    top_p=request.top_p, speed=request.speed,
                )
                return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav", headers={"X-Language": request.language})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to generate audio: {e}")

        return web_app

    @modal.method()
    def generate(self, text: str, speaker_wav: str, language: str = "tr",
                 temperature: float = 0.75, length_penalty: float = 1.0,
                 repetition_penalty: float = 5.0, top_k: int = 50,
                 top_p: float = 0.85, speed: float = 1.0) -> bytes:
        wav = self.tts.tts(
            text=text, speaker_wav=speaker_wav, language=language,
            temperature=temperature, length_penalty=length_penalty,
            repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p, speed=speed,
        )
        wav_tensor = torch.tensor(wav).unsqueeze(0)
        buffer = io.BytesIO()
        ta.save(buffer, wav_tensor, self.tts.synthesizer.output_sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()


@app.local_entrypoint()
def test(
    prompt: str = "Merhaba! Bu, Türkçe ses sentezi testidir.",
    voice_key: str = "voices/system/default.wav",
    language: str = "tr",
    output_path: str = "C:/Users/USER/resolabs/output.wav",
    temperature: float = 0.75,
    speed: float = 1.0,
):
    import pathlib
    xtts = XTTS()
    print(f"Generating '{language}' speech for: {prompt[:60]}...")
    audio_bytes = xtts.generate.remote(
        text=prompt, speaker_wav=f"{R2_MOUNT_PATH}/{voice_key}",
        language=language, temperature=temperature, speed=speed,
    )
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(audio_bytes)
    print(f"Audio saved to {output_file}")