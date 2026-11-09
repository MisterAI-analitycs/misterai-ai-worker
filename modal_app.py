"""MisterAI worker en Modal.

- process_match: función con GPU que analiza un partido (ver processor.py).
- api: endpoint HTTPS que el backend de Render llama al terminar una subida.

Despliegue:  modal deploy modal_app.py
Prueba:      modal run modal_app.py --partido-id <uuid>
"""
import modal

app = modal.App("misterai-worker")

# Los pesos de YOLO se descargan al construir la imagen, no en cada arranque.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("ultralytics", "opencv-python-headless", "numpy", "boto3", "httpx", "fastapi[standard]")
    .run_commands(
        "mkdir -p /models && cd /models && python -c \"from ultralytics import YOLO; YOLO('yolov8m.pt')\""
    )
    .add_local_python_source("processor")
)

# Secret con: SUPABASE_URL, SUPABASE_SECRET_KEY, R2_ENDPOINT, R2_BUCKET,
# R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY y WORKER_TOKEN.
secrets = [modal.Secret.from_name("misterai-secrets")]


# scaledown_window=2: el contenedor (y su GPU) se apaga 2 s después de terminar. Por defecto espera
# 60 s en reposo y esos segundos se facturan.
@app.function(image=image, gpu="A10G", cpu=4, memory=16384, secrets=secrets, timeout=60 * 60,
              retries=0, scaledown_window=2)
def process_match(partido_id: str):
    import processor

    processor.run(partido_id)


@app.function(image=image, secrets=secrets)
@modal.asgi_app()
def api():
    import hmac
    import os
    import re

    from fastapi import FastAPI, Header, HTTPException

    web = FastAPI()
    uuid_re = re.compile(r"^[0-9a-fA-F-]{36}$")

    @web.get("/health")
    def health():
        return {"ok": True}

    @web.post("/procesar")
    def procesar(payload: dict, x_worker_token: str = Header(default="")):
        if not hmac.compare_digest(x_worker_token, os.environ["WORKER_TOKEN"]):
            raise HTTPException(401, "No autorizado")
        partido_id = str(payload.get("partido_id", ""))
        if not uuid_re.match(partido_id):
            raise HTTPException(400, "partido_id no válido")
        process_match.spawn(partido_id)  # no esperamos: el análisis dura minutos
        return {"ok": True}

    return web


@app.local_entrypoint()
def main(partido_id: str):
    process_match.remote(partido_id)
