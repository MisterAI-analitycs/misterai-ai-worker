# misterai-ai-worker

Worker de IA de MisterAI en Modal: analiza un partido (YOLO + ByteTrack), calcula mapas de calor y estadísticas,
guarda el resultado en Supabase y borra el vídeo de R2.

## Puesta en marcha
1. `pip install modal` y `modal setup` (inicia sesión en tu cuenta).
2. Crea el secreto `misterai-secrets` en el panel de Modal (*Secrets*) con: `SUPABASE_URL`, `SUPABASE_SECRET_KEY`,
   `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` y `WORKER_TOKEN`
   (una cadena aleatoria larga; el backend la enviará en la cabecera `x-worker-token`).
3. `modal deploy modal_app.py`. Al terminar muestra la URL del endpoint (`.../procesar`).
4. Prueba con un partido que ya esté en estado `pendiente`: `modal run modal_app.py --partido-id <uuid>`.

## Pesos propios
Para usar `yolo_futbol.pt`: súbelo a un Volume de Modal, móntalo en `process_match` y define `MODEL_WEIGHTS` con su ruta.

## Variables opcionales
`TARGET_FPS` (5), `IMGSZ` (1280), `MAX_MINUTES` (0 = completo).
