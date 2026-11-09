"""Procesamiento de un partido.

Descarga el vídeo de R2, detecta y sigue jugadores y balón con YOLO + ByteTrack,
calcula mapas de calor y estadísticas, guarda el resultado en Supabase y borra el vídeo.
"""
import json
import os
import tempfile
import time
import traceback

import boto3
import cv2
import httpx
import numpy as np
from botocore.config import Config
from botocore.exceptions import ClientError

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "")

WEIGHTS = os.environ.get("MODEL_WEIGHTS", "/models/yolov8m.pt")  # pesos propios: ruta en un Volume
IMGSZ = int(os.environ.get("IMGSZ", "1280"))
TARGET_FPS = float(os.environ.get("TARGET_FPS", "5"))      # fotogramas por segundo analizados
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "0"))    # 0 = partido completo (la prueba gratis usa 45)
GRID_W, GRID_H = 52, 34
PITCH_W, PITCH_H = 105.0, 68.0
MAX_SPEED_MS = 12.0                                         # por encima, el salto se considera ruido
PERSON_NAMES = {"person", "player", "goalkeeper", "referee", "jugador", "portero", "arbitro", "árbitro"}
BALL_NAMES = {"sports ball", "ball", "balon", "balón"}

MSG_VIDEO_CADUCADO = ("El vídeo ya no está en el servidor (se borra a las 72 horas). "
                      "Vuelve a subirlo para analizarlo.")
MSG_ANALISIS_FALLIDO = "No se pudo completar el análisis. Inténtalo de nuevo; si persiste, contacta con soporte."

# Tarifas de Modal (USD por segundo) y recursos de process_match en modal_app.py.
# Si cambias la GPU, los núcleos o la memoria allí, actualízalos aquí.
TARIFA_GPU = 0.000306            # Nvidia A10
NUCLEOS, TARIFA_CPU = 4, 0.0000131   # por núcleo físico
MEMORIA_GIB, TARIFA_MEM = 16, 0.00000222
TARIFA_SEGUNDO = TARIFA_GPU + NUCLEOS * TARIFA_CPU + MEMORIA_GIB * TARIFA_MEM
SCALEDOWN_S = 2                  # espera en reposo antes de apagar el contenedor


# ---------- Coste ----------
def coste_modal(t_run):
    """Segundos facturables aproximados y coste estimado (USD) de esta ejecución.

    Incluye el arranque del contenedor, medido con la hora de inicio del proceso 1 (acotado a 90 s
    por si el contenedor se reutiliza). Es una estimación: la factura real de Modal puede diferir.
    """
    seg = t_run
    try:
        import psutil
        desde_arranque = time.time() - psutil.Process(1).create_time()
        seg = min(max(t_run, desde_arranque), t_run + 90)
    except Exception:
        pass
    seg += SCALEDOWN_S
    return round(seg, 1), round(seg * TARIFA_SEGUNDO, 5)


def _acumulado(partido, seg, coste):
    """Suma esta ejecución a lo ya gastado en el mismo partido (reintentos incluidos)."""
    return {
        "modal_segundos": round(float(partido.get("modal_segundos") or 0) + seg, 1),
        "modal_coste_usd": round(float(partido.get("modal_coste_usd") or 0) + coste, 5),
    }


# ---------- Supabase y R2 ----------
def _headers(extra=None):
    h = {"apikey": SUPABASE_SECRET_KEY, "Content-Type": "application/json"}
    if SUPABASE_SECRET_KEY.startswith("eyJ"):  # clave legacy service_role (JWT)
        h["Authorization"] = f"Bearer {SUPABASE_SECRET_KEY}"
    if extra:
        h.update(extra)
    return h


def _json_default(o):
    """Permite serializar a JSON los tipos de NumPy (float32, int64, arrays...)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tipo no serializable: {type(o).__name__}")


def get_partido(partido_id):
    r = httpx.get(f"{SUPABASE_URL}/rest/v1/partidos",
                  params={"id": f"eq.{partido_id}", "select": "*", "limit": "1"},
                  headers=_headers(), timeout=20)
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


def patch_partido(partido_id, fields):
    r = httpx.patch(f"{SUPABASE_URL}/rest/v1/partidos", params={"id": f"eq.{partido_id}"},
                    content=json.dumps(fields, default=_json_default),
                    headers=_headers({"Prefer": "return=minimal"}), timeout=60)
    r.raise_for_status()


def r2_client():
    return boto3.client(
        "s3", endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _video_no_encontrado(e):
    """True si el error es que el objeto ya no existe en R2 (borrado por la regla de retención)."""
    return isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound")


def marcar_error(partido_id, codigo, mensaje, detalle=None, quitar_video=False, extra=None):
    """Deja el partido en estado 'error' con un mensaje legible para el panel."""
    resultados = {"codigo": codigo, "error": mensaje}
    if detalle:
        resultados["detalle"] = detalle[:300]  # solo para depuración; el panel no lo muestra
    campos = {"estado_analisis": "error", "resultados": resultados}
    if quitar_video:
        campos["file_path"] = None
    if extra:
        campos.update(extra)
    patch_partido(partido_id, campos)


# ---------- Utilidades de visión ----------
def _color_camiseta(img, box):
    """Color medio (Lab) de la zona del torso, para separar equipos."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None
    ya, yb = max(y1 + int(h * 0.2), 0), max(y1 + int(h * 0.55), 0)
    xa, xb = max(x1 + int(w * 0.2), 0), max(x2 - int(w * 0.2), 0)
    crop = img[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)


def _agrupar_equipos(colores):
    """K-means de 2 grupos sobre el color medio de cada pista. Devuelve {id_pista: 0|1}."""
    if len(colores) < 4:
        return {t: 0 for t in colores}
    data = np.float32(list(colores.values()))
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, etiquetas, _ = cv2.kmeans(data, 2, None, criterios, 5, cv2.KMEANS_PP_CENTERS)
    return dict(zip(colores.keys(), etiquetas.ravel().tolist()))


def _normalizar(grid):
    m = grid.max()
    return (np.round(100 * grid / m).astype(int) if m > 0 else grid).tolist()


# ---------- Análisis ----------
def analizar_video(path, calibracion=None, max_minutes=0.0):
    from ultralytics import YOLO

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if W == 0 or H == 0 or total == 0:
        raise ValueError("No se pudo leer el vídeo")

    stride = max(1, round(fps / TARGET_FPS))
    limit = min(total, int(max_minutes * 60 * fps)) if max_minutes else total

    # Plano: campo real en metros si hay calibración (4 esquinas del campo en el vídeo), si no, imagen normalizada.
    homog = None
    if calibracion and len(calibracion.get("puntos_imagen", [])) == 4:
        src = np.float32(calibracion["puntos_imagen"])
        dst = np.float32([[0, 0], [PITCH_W, 0], [PITCH_W, PITCH_H], [0, PITCH_H]])
        homog = cv2.getPerspectiveTransform(src, dst)
    ext_w, ext_h = (PITCH_W, PITCH_H) if homog is not None else (1.0, 1.0)
    poss_radius = 3.0 if homog is not None else 0.07

    def to_plane(x, y):
        x, y = float(x), float(y)  # los valores del modelo son float32 de NumPy: se pasan a float de Python
        if homog is None:
            return x / W, y / H
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), homog)[0][0]
        return float(p[0]), float(p[1])

    def cell(px, py):
        cx, cy = int(px / ext_w * GRID_W), int(py / ext_h * GRID_H)
        return (cx, cy) if 0 <= cx < GRID_W and 0 <= cy < GRID_H else None

    model = YOLO(WEIGHTS)
    person_ids = {i for i, n in model.names.items() if n.lower() in PERSON_NAMES}
    ball_ids = {i for i, n in model.names.items() if n.lower() in BALL_NAMES}
    if not person_ids:
        raise ValueError("El modelo no tiene una clase de jugador")

    grids, nobs, sumxy, dist, last, muestras, poss = {}, {}, {}, {}, {}, {}, {}
    ball_traj, ball_frames, frames_done, poss_frames = [], 0, 0, 0

    t_inf = time.time()
    stream = model.track(source=path, stream=True, persist=True, tracker="bytetrack.yaml", vid_stride=stride,
                         imgsz=IMGSZ, conf=0.2, classes=sorted(person_ids | ball_ids), half=True, verbose=False)
    for i, r in enumerate(stream):
        frame_idx = i * stride
        if frame_idx >= limit:
            break
        frames_done += 1
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy, cls = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        confs = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else np.full(len(cls), -1)

        ball, players = None, []
        for box, c, cf, tid in zip(xyxy, cls, confs, ids):
            if c in ball_ids:
                if ball is None or cf > ball[2]:
                    ball = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, cf)
            elif tid >= 0:
                players.append((int(tid), (box[0] + box[2]) / 2, box[3], box))

        t_s = frame_idx / fps
        pos_pl = {}
        for tid, fx, fy, box in players:
            px, py = to_plane(fx, fy)
            pos_pl[tid] = (px, py)
            nobs[tid] = nobs.get(tid, 0) + 1
            s = sumxy.setdefault(tid, [0.0, 0.0])
            s[0] += px
            s[1] += py
            ce = cell(px, py)
            if ce:
                if tid not in grids:
                    grids[tid] = np.zeros((GRID_H, GRID_W), np.int32)
                grids[tid][ce[1], ce[0]] += 1
            if homog is not None:
                prev = last.get(tid)
                if prev:
                    d = ((px - prev[0]) ** 2 + (py - prev[1]) ** 2) ** 0.5
                    if d / max(t_s - prev[2], 1e-3) <= MAX_SPEED_MS:
                        dist[tid] = dist.get(tid, 0.0) + d
                last[tid] = (px, py, t_s)
            sm = muestras.setdefault(tid, [])
            if len(sm) < 25 and nobs[tid] % 8 == 1:
                col = _color_camiseta(r.orig_img, box)
                if col is not None:
                    sm.append(col)

        if ball:
            ball_frames += 1
            bx, by = to_plane(ball[0], ball[1])
            if ball_frames % 5 == 1 and len(ball_traj) < 3000:
                ball_traj.append([round(t_s, 1), round(bx, 3), round(by, 3)])
            if pos_pl:
                tid, (px, py) = min(pos_pl.items(), key=lambda kv: (kv[1][0] - bx) ** 2 + (kv[1][1] - by) ** 2)
                if ((px - bx) ** 2 + (py - by) ** 2) ** 0.5 <= poss_radius:
                    poss[tid] = poss.get(tid, 0) + 1
                    poss_frames += 1
    t_inf = time.time() - t_inf

    if frames_done == 0:
        raise ValueError("No se analizó ningún fotograma")

    # Equipos por color de camiseta (solo pistas estables).
    min_obs = max(20, int(0.05 * max(nobs.values(), default=0)))
    elegibles = {t: np.mean(c, axis=0) for t, c in muestras.items() if len(c) >= 3 and nobs.get(t, 0) >= min_obs}
    equipo_de = _agrupar_equipos(elegibles)

    equipos = {}
    for idx, nombre in enumerate(("a", "b")):
        tids = [t for t, e in equipo_de.items() if e == idx]
        grid = sum((grids[t] for t in tids if t in grids), np.zeros((GRID_H, GRID_W), np.int32))
        posesion = sum(poss.get(t, 0) for t in tids)
        equipos[nombre] = {
            "pistas": len(tids),
            "mapa_calor": _normalizar(grid),
            "posesion_pct": round(100 * posesion / poss_frames, 1) if poss_frames else None,
        }

    jugadores = []
    for tid in sorted(elegibles, key=lambda t: -nobs[t])[:30]:
        n = nobs[tid]
        jugadores.append({
            "id": tid, "equipo": "ab"[equipo_de[tid]], "observaciones": n,
            "posicion_media": [round(sumxy[tid][0] / n, 3), round(sumxy[tid][1] / n, 3)],
            "distancia_m": round(dist[tid], 1) if tid in dist else None,
        })

    return {
        "version": 1,
        "plano": "campo" if homog is not None else "imagen",
        "campo_m": [PITCH_W, PITCH_H] if homog is not None else None,
        "rejilla": {"ancho": GRID_W, "alto": GRID_H},
        "equipos": equipos,
        "jugadores": jugadores,
        "balon": {"detectado_pct": round(100 * ball_frames / frames_done, 1), "trayectoria": ball_traj},
        "metricas": {
            "duracion_video_s": round(total / fps, 1), "segundos_analizados": round(frames_done * stride / fps, 1),
            "fps_origen": round(fps, 2), "fps_proceso": round(fps / stride, 2), "frames_procesados": frames_done,
            "resolucion": [W, H], "modelo": os.path.basename(WEIGHTS), "tiempo_inferencia_s": round(t_inf, 1),
        },
    }


def run(partido_id, max_minutes=None):
    """Punto de entrada: procesa un partido de principio a fin."""
    t0 = time.time()
    partido = get_partido(partido_id)
    if not partido:
        print(f"Partido {partido_id} no encontrado")
        return
    if partido["estado_analisis"] not in ("pendiente", "error"):
        print(f"Partido {partido_id} ya está en estado {partido['estado_analisis']}; no se repite")
        return

    key = partido.get("file_path")
    if not key:  # sin vídeo no hay nada que analizar
        marcar_error(partido_id, "video_caducado", MSG_VIDEO_CADUCADO)
        return

    patch_partido(partido_id, {"estado_analisis": "procesando"})
    s3 = r2_client()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "video.mp4")
            t_dl = time.time()
            s3.download_file(R2_BUCKET, key, path)
            t_dl = time.time() - t_dl
            resultados = analizar_video(path, partido.get("calibracion"),
                                        MAX_MINUTES if max_minutes is None else max_minutes)
        t_run = time.time() - t0
        seg, coste = coste_modal(t_run)
        resultados["metricas"]["tiempo_descarga_s"] = round(t_dl, 1)
        resultados["metricas"]["tiempo_total_s"] = round(t_run, 1)
        # Privacidad: el vídeo se borra en cuanto termina el análisis.
        patch_partido(partido_id, {"estado_analisis": "completado", "resultados": resultados, "file_path": None,
                                   **_acumulado(partido, seg, coste)})
        try:
            s3.delete_object(Bucket=R2_BUCKET, Key=key)
        except Exception:
            traceback.print_exc()  # si falla, la regla de ciclo de vida de R2 lo borra
        print(f"Partido {partido_id} completado en {round(t_run, 1)} s (coste Modal estimado: {coste} USD)")
    except Exception as e:
        traceback.print_exc()
        seg, coste = coste_modal(time.time() - t0)  # un análisis fallido también consume computación
        extra = _acumulado(partido, seg, coste)
        if _video_no_encontrado(e):
            # El vídeo ya no existe en R2: se deja constancia y se quita la referencia al archivo.
            marcar_error(partido_id, "video_caducado", MSG_VIDEO_CADUCADO, str(e), quitar_video=True, extra=extra)
        else:
            marcar_error(partido_id, "analisis_fallido", MSG_ANALISIS_FALLIDO, str(e), extra=extra)
