"""XTTS v2 API - Text-to-speech with Auto-Language Detection on Modal."""

import modal
import os

# R2 cloud bucket mount (read-only)
R2_BUCKET_NAME = "resolabs-app"
R2_ACCOUNT_ID = "de31186ed80914fd17e2219c23eefa03"
R2_MOUNT_PATH = "/r2"
r2_bucket = modal.CloudBucketMount(
    R2_BUCKET_NAME,
    bucket_endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    secret=modal.Secret.from_name("cloudflare-r2"),
    read_only=True,
)

# Modal setup: Added langdetect
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "TTS==0.22.0",
        "fastapi[standard]==0.124.4",
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "transformers==4.37.2",
        "langdetect==1.0.9", # Dil algılama kütüphanesi eklendi
    )
)
app = modal.App("xtts-api", image=image)

with image.imports():
    import io
    from pathlib import Path
    import torch
    import torchaudio as ta
    from fastapi import Depends, FastAPI, HTTPException, Security
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field

    api_key_scheme = APIKeyHeader(
        name="x-api-key",
        scheme_name="ApiKeyAuth",
        auto_error=False,
    )

    def verify_api_key(x_api_key: str | None = Security(api_key_scheme)):
        expected = os.environ.get("CHATTERBOX_API_KEY", "")
        if not expected or x_api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return x_api_key

    class TTSRequest(BaseModel):
        """Request model for text-to-speech generation."""
        prompt: str = Field(..., min_length=1, max_length=5000)
        voice_key: str = Field(..., min_length=1, max_length=300)
        # Dil artık varsayılan olarak "auto" geliyor
        language: str = Field(default="auto", min_length=2, max_length=5) 
        speed: float = Field(default=1.0, ge=0.5, le=2.0)


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("chatterbox-api-key"),
        modal.Secret.from_name("cloudflare-r2"),
    ],
    volumes={R2_MOUNT_PATH: r2_bucket},
)
@modal.concurrent(max_inputs=10)
class XTTSGenerator:
    @modal.enter()
    def load_model(self):
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    @modal.asgi_app()
    def serve(self):
        web_app = FastAPI(
            title="XTTS Voice Cloning API",
            description="High-quality Text-to-speech with auto-language detection.",
            docs_url="/docs",
            dependencies=[Depends(verify_api_key)],
        )
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.post("/generate", responses={200: {"content": {"audio/wav": {}}}})
        def generate_speech(request: TTSRequest):
            voice_path = Path(R2_MOUNT_PATH) / request.voice_key
            if not voice_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice not found at '{request.voice_key}'",
                )

            try:
                audio_bytes = self.generate.local(
                    request.prompt,
                    str(voice_path),
                    request.language,
                    request.speed,
                )
                return StreamingResponse(
                    io.BytesIO(audio_bytes),
                    media_type="audio/wav",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate audio: {e}",
                )

        return web_app

    @modal.method()
    def generate(
        self,
        prompt: str,
        audio_prompt_path: str,
        language: str = "auto",
        speed: float = 1.0,
    ):
        # Eğer dil 'auto' olarak geldiyse, metni analiz et
        if language == "auto":
            from langdetect import detect
            try:
                detected_lang = detect(prompt)
                # XTTS'nin desteklediği başlıca dillerin listesi
                supported_langs = ["en", "tr", "de", "fr", "es", "it", "pt", "pl", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
                
                if detected_lang in supported_langs:
                    language = detected_lang
                    print(f"Dil otomatik algılandı: {language}")
                else:
                    language = "tr" # Desteklenmeyen bir dilse Türkçeye dön (fallback)
            except:
                language = "tr" # Algılama başarısız olursa Türkçeye dön

        wav_data = self.model.tts(
            text=prompt,
            speaker_wav=audio_prompt_path,
            language=language,
            speed=speed
        )

        wav_tensor = torch.tensor(wav_data).unsqueeze(0)
        sample_rate = 24000

        buffer = io.BytesIO()
        ta.save(buffer, wav_tensor, sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()


@app.local_entrypoint()
def test(
    prompt: str = "Hello, how are you today? This is an English test.",
    voice_key: str = "voices/system/default.wav",
    output_path: str = "/tmp/xtts-api/output.wav",
    language: str = "auto",
):
    import pathlib

    generator = XTTSGenerator()
    audio_prompt_path = f"{R2_MOUNT_PATH}/{voice_key}"
    
    audio_bytes = generator.generate.remote(
        prompt=prompt,
        audio_prompt_path=audio_prompt_path,
        language=language,
    )

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(audio_bytes)
    print(f"Audio saved to {output_file}")