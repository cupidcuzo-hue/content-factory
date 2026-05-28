"""
Cuzo Content Factory — Python Backend
  Images → Laozhang.ai (nano-banana-pro / seedream)
  Videos → KIE.ai (Kling 2.6 / 3.0 / v2-1)
  Cost tracking → SQLite (with /tmp fallback)
  Real-time progress → Socket.io
  Storage → Google Drive
"""

from __future__ import annotations
import eventlet
eventlet.monkey_patch()
import os, sys, json, time, io, threading, sqlite3, datetime, logging
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO

# ── Startup env checks ────────────────────────────────────────────────────────

LAOZHANG_API_KEY = os.environ.get('LAOZHANG_API_KEY', '')
KIE_API_KEY      = os.environ.get('KIE_API_KEY', '')

# Warn on missing keys but don't crash — app still starts and serves the frontend
_missing = [k for k, v in [('LAOZHANG_API_KEY', LAOZHANG_API_KEY), ('KIE_API_KEY', KIE_API_KEY)] if not v]
if _missing:
    import warnings
    warnings.warn(f"⚠️  Missing env vars: {', '.join(_missing)} — those API calls will fail at runtime.")

GOOGLE_SA_JSON       = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON', '')
GOOGLE_DRIVE_FOLDER  = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', '')
DB_PATH              = os.environ.get('DB_PATH', 'data/content_factory.db')

# ── Flask / SocketIO ──────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cuzo-cf-secret-2025')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Database ──────────────────────────────────────────────────────────────────

_DB_PATH = DB_PATH  # may be overridden if preferred path fails

def get_db():
    db_dir = os.path.dirname(_DB_PATH)
    if db_dir:
        try:
            os.makedirs(db_dir, exist_ok=True)
        except OSError as e:
            log.warning(f"Cannot create DB directory {db_dir}: {e}")
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    global _DB_PATH
    try:
        with get_db() as db:
            db.execute('''
                CREATE TABLE IF NOT EXISTS cost_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id       TEXT,
                    provider     TEXT,
                    model        TEXT,
                    content_type TEXT,
                    model_name   TEXT,
                    quantity     INTEGER DEFAULT 1,
                    cost_per_unit REAL,
                    total_cost   REAL,
                    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            db.commit()
        log.info(f"Database ready at {_DB_PATH}")
    except Exception as e:
        # Fall back to an in-process temp DB rather than crashing the server
        log.warning(f"DB init failed at {_DB_PATH} ({e}) — falling back to /tmp/content_factory.db")
        _DB_PATH = '/tmp/content_factory.db'
        try:
            with get_db() as db:
                db.execute('''
                    CREATE TABLE IF NOT EXISTS cost_log (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id       TEXT, provider TEXT, model TEXT, content_type TEXT,
                        model_name   TEXT, quantity INTEGER DEFAULT 1,
                        cost_per_unit REAL, total_cost REAL,
                        generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                db.commit()
            log.info(f"Database ready (fallback) at {_DB_PATH}")
        except Exception as e2:
            log.error(f"Fallback DB also failed: {e2} — cost tracking disabled")

init_db()

# ── Cost constants ────────────────────────────────────────────────────────────

LAOZHANG_COSTS = {'1K': 0.0125, '2K': 0.025, '4K': 0.050}

# Per-model cost overrides for Laozhang (for models that aren't resolution-priced)
LAOZHANG_MODEL_COSTS = {
    'gpt-image-1':             0.040,
    'gpt-image-2':             0.040,
    'gpt-image-2-all':         0.040,
    'gpt-image-2-vip':         0.060,
    'gemini-2.5-flash-image':  0.020,
    'gemini-3-pro-image-preview': 0.060,
    'gemini-3.1-flash-image-preview': 0.025,
    'dall-e-3':                0.080,
    'grok-2-aurora':           0.070,
    'imagen-3.0-generate-002': 0.050,
    'flux-1.1-pro':            0.050,
    'flux-1-dev':              0.030,
    'flux-1-schnell':          0.010,
}

KIE_IMAGE_COSTS = {
    # NanoBanana (Google Gemini-based) — confirmed working
    'nano-banana-pro':              0.040,
    'nano-banana-2':                0.040,
    'google/nano-banana':           0.020,
    # GPT Image via KIE
    'gpt-image-2-text-to-image':    0.040,
    # Grok Imagine via KIE
    'grok-imagine/text-to-image':   0.060,
    # Google Imagen 4 via KIE
    'google/imagen4':               0.050,
    'google/imagen4-ultra':         0.080,
}

VIDEO_COSTS = {
    'kling/v2-1-standard':      0.125,
    'kling/v2-1-pro':           0.250,
    'kling/v2-1-master':        0.800,
    'kling-3.0/video':          0.350,
    'kling-2.6/text-to-video':  0.250,
    'kling-3.0/motion-control': 0.500,
    'kling-2.6/motion-control': 0.350,
}

LAOZHANG_VIDEO_COSTS = {
    'wan2.1-14b-720p':      0.15,
    'wan2.1-14b-480p':      0.08,
    'wan2.1-i2v-14b-720p':  0.18,
    # Kling models accessible via Laozhang
    'kling-3.0/video':          0.350,
    'kling-2.6/text-to-video':  0.250,
    'kling/v2-1-pro':           0.250,
    'kling/v2-1-standard':      0.125,
    'kling/v2-1-master':        0.800,
}

LAOZHANG_VIDEO_SIZE_MAP = {
    '9:16': '720x1280',
    '16:9': '1280x720',
    '1:1':  '720x720',
    '3:4':  '576x768',
    '4:3':  '768x576',
}

IMG_SIZE_MAP = {
    ('9:16', '2K'): '1024x1820', ('9:16', '4K'): '2048x3640', ('9:16', '1K'): '512x910',
    ('1:1',  '2K'): '1024x1024', ('1:1',  '4K'): '2048x2048', ('1:1',  '1K'): '512x512',
    ('4:3',  '2K'): '1024x768',  ('4:3',  '4K'): '2048x1536', ('4:3',  '1K'): '512x384',
    ('3:4',  '2K'): '768x1024',  ('3:4',  '4K'): '1536x2048', ('3:4',  '1K'): '384x512',
    ('16:9', '2K'): '1820x1024', ('16:9', '4K'): '3640x2048', ('16:9', '1K'): '910x512',
}

# DALL-E 3 only supports these exact sizes
DALLE3_SIZE_MAP = {
    '9:16': '1024x1792', '3:4': '1024x1792', '1:1': '1024x1024',
    '4:3': '1792x1024', '16:9': '1792x1024',
}

# gpt-image-1 only supports these exact sizes
GPTI1_SIZE_MAP = {
    '9:16': '1024x1536', '3:4': '1024x1536', '1:1': '1024x1024',
    '4:3': '1536x1024', '16:9': '1536x1024',
}

# gpt-image-2 / gemini require sizes that are multiples of 16
GPTI2_SIZE_MAP = {
    '9:16': '1024x1792', '3:4': '1024x1344', '1:1': '1024x1024',
    '4:3': '1344x1024', '16:9': '1792x1024',
}

# Models that need special parameter handling
DALLE3_MODELS   = {'dall-e-3'}
GPTI1_MODELS    = {'gpt-image-1'}
GPTI2_MODELS    = {'gpt-image-2', 'gpt-image-2-all', 'gpt-image-2-vip',
                   'gemini-2.5-flash-image', 'gemini-3-pro-image-preview',
                   'gemini-3.1-flash-image-preview'}

def _lz_payload(model: str, prompt: str, ratio: str, resolution: str, ref_urls: list, n: int = 1) -> dict:
    """Build correct Laozhang payload for any model — handles size/quality differences."""
    if model in DALLE3_MODELS:
        size = DALLE3_SIZE_MAP.get(ratio, '1024x1792')
        p = {'model': model, 'prompt': prompt, 'n': n, 'size': size, 'quality': 'hd'}
    elif model in GPTI1_MODELS:
        size = GPTI1_SIZE_MAP.get(ratio, '1024x1536')
        p = {'model': model, 'prompt': prompt, 'n': n, 'size': size, 'quality': 'high'}
    elif model in GPTI2_MODELS:
        size = GPTI2_SIZE_MAP.get(ratio, '1024x1792')
        p = {'model': model, 'prompt': prompt, 'n': n, 'size': size, 'quality': 'standard'}
    else:
        size = IMG_SIZE_MAP.get((ratio, resolution), '1024x1820')
        p = {'model': model, 'prompt': prompt, 'n': n, 'size': size, 'quality': 'hd'}
    if ref_urls and model not in DALLE3_MODELS and model not in GPTI1_MODELS and model not in GPTI2_MODELS:
        p['image_input'] = ref_urls
    return p

KIE_TIER_MAP = {
    'standard': 'kling/v2-1-standard',
    'pro':      'kling/v2-1-pro',
    'master':   'kling/v2-1-master',
}

KIE_BASE = 'https://api.kie.ai/api/v1/jobs'

# ── Cost helpers ──────────────────────────────────────────────────────────────

def log_cost(job_id, provider, model, content_type, model_name, cost_per_unit, quantity=1):
    total = round(cost_per_unit * quantity, 6)
    with get_db() as db:
        db.execute(
            'INSERT INTO cost_log (job_id,provider,model,content_type,model_name,quantity,cost_per_unit,total_cost) VALUES (?,?,?,?,?,?,?,?)',
            (job_id, provider, model, content_type, model_name, quantity, cost_per_unit, total)
        )
        db.commit()
    return total

def today_total():
    today = datetime.date.today().isoformat()
    with get_db() as db:
        row = db.execute(
            "SELECT COALESCE(SUM(total_cost),0) as t FROM cost_log WHERE date(generated_at)=?", (today,)
        ).fetchone()
    return float(row['t']) if row else 0.0

# ── Image cache (b64 responses like gpt-image-2 / Gemini) ────────────────────
# Primary: write to /tmp/cuzo_imgs/ so images survive in-process and on-disk.
# Fallback in-memory dict kept for backwards compat.
_IMG_DIR = '/tmp/cuzo_imgs'
os.makedirs(_IMG_DIR, exist_ok=True)
_img_cache: dict[str, bytes] = {}

def cache_img(job_id: str, data_bytes: bytes) -> str:
    """Persist image bytes to disk + memory and return a server-relative URL."""
    _img_cache[job_id] = data_bytes
    try:
        path = os.path.join(_IMG_DIR, f'{job_id}.png')
        with open(path, 'wb') as f:
            f.write(data_bytes)
    except Exception as ex:
        log.warning(f'cache_img disk write failed [{job_id}]: {ex}')
    return f'/api/img/{job_id}'

@app.route('/api/img/<job_id>')
def serve_cached_img(job_id):
    # 1. Try in-memory cache (fastest)
    data = _img_cache.get(job_id)
    if data:
        return send_file(io.BytesIO(data), mimetype='image/png')
    # 2. Try disk (survives server restart within same Railway instance)
    path = os.path.join(_IMG_DIR, f'{job_id}.png')
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return jsonify(error='Image not found — server may have restarted'), 404

# ── Socket.io emit helper ──────────────────────────────────────────────────────

def emit_to(socket_id, event, data):
    """Emit a Socket.io event to a specific client room."""
    if socket_id:
        socketio.emit(event, data, room=socket_id)
    else:
        socketio.emit(event, data)  # broadcast if no room

# ── Google Drive upload ───────────────────────────────────────────────────────

def upload_to_drive(data: bytes, filename: str, mime_type: str) -> str:
    """Upload bytes to Google Drive; return shareable URL or empty string."""
    if not GOOGLE_SA_JSON or not GOOGLE_DRIVE_FOLDER:
        log.warning("Google Drive not configured (GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_DRIVE_FOLDER_ID missing) — skipping upload")
        return ''
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload

        creds_info = json.loads(GOOGLE_SA_JSON)
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)

        file_meta = {'name': filename, 'parents': [GOOGLE_DRIVE_FOLDER]}
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
        f = service.files().create(body=file_meta, media_body=media, fields='id').execute()
        fid = f.get('id')
        service.permissions().create(fileId=fid, body={'type': 'anyone', 'role': 'reader'}).execute()
        url = f'https://drive.google.com/file/d/{fid}/view'
        log.info(f"Uploaded to Drive: {filename} → {url}")
        return url
    except Exception as e:
        log.error(f"Drive upload failed ({filename}): {e}")
        return ''

# ── Image generation via Laozhang.ai ─────────────────────────────────────────

IMG_GEN_TIMEOUT = 90  # seconds — eventlet.Timeout kills hung requests reliably

def gen_image(job_id: str, prompt: str, resolution: str, ratio: str,
              model_name: str, ref_urls: list, socket_id: str, model: str = 'nano-banana-pro'):
    """Generate image via Laozhang, emit result immediately, upload to Drive in background."""
    size = IMG_SIZE_MAP.get((ratio, resolution), '1024x1820')
    cost_per = LAOZHANG_MODEL_COSTS.get(model, LAOZHANG_COSTS.get(resolution, 0.025))

    emit_to(socket_id, 'job:progress', {'job_id': job_id, 'status': 'Generating…', 'pct': 10})

    payload = _lz_payload(model, prompt, ratio, resolution, ref_urls, n=1)

    try:
        with eventlet.Timeout(IMG_GEN_TIMEOUT):
            r = requests.post(
                'https://api.laozhang.ai/v1/images/generations',
                headers={'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'},
                json=payload, timeout=IMG_GEN_TIMEOUT,
            )
        r.raise_for_status()
        data = r.json()
        log.info(f"Laozhang image response [{job_id}]: {json.dumps(data)[:300]}")
        item = data['data'][0]
        img_url = item.get('url')
        cached_bytes = None
        if not img_url and item.get('b64_json'):
            import base64
            cached_bytes = base64.b64decode(item['b64_json'])
            img_url = cache_img(job_id, cached_bytes)
            log.info(f"Laozhang b64 cached [{job_id}] → {img_url}")
        if not img_url:
            raise Exception(f"No URL or b64 in response: {json.dumps(data)[:200]}")
        log.info(f"Laozhang image ready [{job_id}]: {img_url[:60]}…")
    except eventlet.Timeout:
        log.error(f"Laozhang image timed out [{job_id}] after {IMG_GEN_TIMEOUT}s")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': f'Timed out after {IMG_GEN_TIMEOUT}s — API overloaded, try again'})
        return
    except Exception as e:
        body = ''
        try:
            if hasattr(e, 'response') and e.response is not None:
                body = e.response.text[:300]
        except Exception:
            pass
        log.error(f"Laozhang image gen failed [{job_id}]: {e} | body={body}")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': f'Laozhang error: {body or str(e)}'})
        return

    total_cost = log_cost(job_id, 'laozhang', model, 'image', model_name, cost_per)
    emit_to(socket_id, 'job:complete', {'job_id': job_id, 'url': img_url, 'cost': total_cost, 'provider': 'laozhang'})
    emit_to(socket_id, 'cost:update', {'today_total': today_total()})
    log.info(f"Image job complete [{job_id}] cost=${total_cost:.4f}")

    if GOOGLE_SA_JSON and GOOGLE_DRIVE_FOLDER:
        date_str = datetime.date.today().strftime('%Y%m%d')
        safe_name = model_name.replace(' ', '_').replace('/', '_').lower()
        ext = 'png' if cached_bytes else 'jpg'
        filename = f"{safe_name}_image_{date_str}_{job_id[:8]}.{ext}"
        mime = 'image/png' if cached_bytes else 'image/jpeg'
        def _bg_upload(url=img_url, fname=filename, data_bytes=cached_bytes, m=mime):
            try:
                img_data = data_bytes if data_bytes else requests.get(url, timeout=60).content
                upload_to_drive(img_data, fname, m)
            except Exception as ex:
                log.warning(f"BG Drive upload failed [{job_id}]: {ex}")
        threading.Thread(target=_bg_upload, daemon=True).start()


def gen_image_batch(job_ids: list, prompt: str, resolution: str, ratio: str,
                    model_name: str, ref_urls: list, socket_id: str, model: str = 'nano-banana-pro'):
    """One Laozhang call with n=len(job_ids) — fastest possible batch image generation."""
    n = len(job_ids)
    size = IMG_SIZE_MAP.get((ratio, resolution), '1024x1820')
    cost_per = LAOZHANG_MODEL_COSTS.get(model, LAOZHANG_COSTS.get(resolution, 0.025))

    for jid in job_ids:
        emit_to(socket_id, 'job:progress', {'job_id': jid, 'status': f'Generating {n} images…', 'pct': 10})

    payload = _lz_payload(model, prompt, ratio, resolution, ref_urls, n=n)

    batch_timeout = IMG_GEN_TIMEOUT + n * 30  # extra time per image in batch
    try:
        with eventlet.Timeout(batch_timeout):
            r = requests.post(
                'https://api.laozhang.ai/v1/images/generations',
                headers={'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'},
                json=payload, timeout=batch_timeout,
            )
        r.raise_for_status()
        data = r.json()
        log.info(f"Laozhang batch response [{job_ids[0]}]: {json.dumps(data)[:300]}")
        img_urls = []
        for idx, item in enumerate(data['data']):
            u = item.get('url')
            if not u and item.get('b64_json'):
                import base64
                img_bytes = base64.b64decode(item['b64_json'])
                jid = job_ids[idx] if idx < len(job_ids) else f'{job_ids[0]}-{idx}'
                u = cache_img(jid, img_bytes)
            img_urls.append(u or '')
        log.info(f"Laozhang batch ready [{job_ids[0]}…]: {n} images")
    except eventlet.Timeout:
        log.error(f"Laozhang batch timed out after {batch_timeout}s")
        for jid in job_ids:
            emit_to(socket_id, 'job:failed', {'job_id': jid, 'error': f'Timed out after {batch_timeout}s — API overloaded, try again'})
        return
    except Exception as e:
        body = ''
        try:
            if hasattr(e, 'response') and e.response is not None:
                body = e.response.text[:300]
        except Exception:
            pass
        log.error(f"Laozhang batch gen failed: {e} | body={body}")
        for jid in job_ids:
            emit_to(socket_id, 'job:failed', {'job_id': jid, 'error': f'Laozhang error: {body or str(e)}'})
        return

    date_str = datetime.date.today().strftime('%Y%m%d')
    safe_name = model_name.replace(' ', '_').lower()

    # Build (job_id, url, cached_bytes) triples for emit + Drive upload
    triples = []
    for idx, (job_id, img_url) in enumerate(zip(job_ids, img_urls)):
        # Recover cached bytes if this was a b64 image (url starts with /api/img/)
        b64_bytes = _img_cache.get(job_id) if img_url.startswith('/api/img/') else None
        triples.append((job_id, img_url, b64_bytes))

    for job_id, img_url, _ in triples:
        total_cost = log_cost(job_id, 'laozhang', model, 'image', model_name, cost_per)
        emit_to(socket_id, 'job:complete', {
            'job_id': job_id, 'url': img_url, 'cost': total_cost, 'provider': 'laozhang'
        })
        log.info(f"Batch image complete [{job_id}] cost=${total_cost:.4f}")

    emit_to(socket_id, 'cost:update', {'today_total': today_total()})

    # Drive uploads in background — use cached bytes for b64 images to avoid relative-URL issue
    if GOOGLE_SA_JSON and GOOGLE_DRIVE_FOLDER:
        def _bg_batch(t=triples):
            for jid, url, data_bytes in t:
                try:
                    is_b64 = data_bytes is not None
                    ext  = 'png' if is_b64 else 'jpg'
                    mime = 'image/png' if is_b64 else 'image/jpeg'
                    img_data = data_bytes if is_b64 else requests.get(url, timeout=60).content
                    upload_to_drive(img_data, f"{safe_name}_image_{date_str}_{jid[:8]}.{ext}", mime)
                except Exception as ex:
                    log.warning(f"BG Drive upload failed [{jid}]: {ex}")
        threading.Thread(target=_bg_batch, daemon=True).start()


# ── Video generation via KIE.ai ───────────────────────────────────────────────

def _kie_submit_and_poll(job_id, kie_model, payload_input, socket_id, headers):
    """Submit to KIE and poll until done. Returns (url_or_None, error_or_None)."""
    emit_to(socket_id, 'job:progress', {'job_id': job_id, 'status': 'Submitting to KIE…', 'pct': 5})
    try:
        r = requests.post(
            f'{KIE_BASE}/createTask',
            headers=headers,
            json={'model': kie_model, 'input': payload_input},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        if d.get('code') != 200:
            raise Exception(d.get('msg', 'KIE API error'))
        task_id = d['data']['taskId']
        log.info(f"KIE task created [{job_id}]: {task_id}")
    except Exception as e:
        return None, str(e)

    deadline = time.time() + 300
    poll_num = 0
    while time.time() < deadline:
        time.sleep(3 if poll_num < 20 else 5)
        try:
            pr = requests.get(
                f'{KIE_BASE}/recordInfo?taskId={task_id}', headers=headers, timeout=15
            )
            pr.raise_for_status()
            pd = pr.json()
            # Unpack resultJson if it's a string
            if isinstance(pd['data'].get('resultJson'), str):
                try:
                    pd['data']['result'] = json.loads(pd['data']['resultJson'])
                except Exception:
                    pass
            state = pd['data'].get('state')
            poll_num += 1
            pct = min(10 + poll_num * 3, 88)
            emit_to(socket_id, 'job:progress', {
                'job_id': job_id, 'status': f'Processing… ({state})', 'pct': pct
            })
            if state == 'success':
                res = pd['data'].get('result', {})
                urls = (res.get('resultUrls')
                        or [v['url'] for v in res.get('videos', [])]
                        or ([res['url']] if res.get('url') else []))
                if urls:
                    return urls[0], None
            elif state == 'fail':
                return None, 'KIE generation failed'
        except Exception as e:
            log.warning(f"Poll error [{job_id}]: {e}")
    return None, 'Timed out after 300s'


def gen_video(job_id: str, prompt: str, model: str, duration: str, ratio: str,
              image_url: str | None, mode: str | None, model_name: str, socket_id: str,
              mc_input_urls: list | None = None, mc_video_urls: list | None = None,
              mc_orientation: str | None = None):
    """Generate video via KIE.ai, upload to Drive, log cost.
    Motion control mode: pass mc_input_urls + mc_video_urls instead of prompt/duration.
    """
    # Resolve model string (tier shorthand or direct KIE model string)
    kie_model = KIE_TIER_MAP.get(model, model)
    cost_per = VIDEO_COSTS.get(kie_model, VIDEO_COSTS.get(model, 0.250))

    is_motion_control = mc_input_urls or mc_video_urls

    if is_motion_control:
        # Motion control: different input structure
        payload_input = {
            'input_urls': mc_input_urls or [],
            'video_urls': mc_video_urls or [],
            'mode': mode or 'std',
        }
        if mc_orientation and 'kling-3.0' not in kie_model:
            payload_input['character_orientation'] = mc_orientation
    else:
        is_kling3 = 'kling-3.0' in kie_model or 'kling/v2-1' in kie_model
        safe_dur = duration if is_kling3 else ('5' if int(duration or 5) <= 5 else '10')

        payload_input = {
            'prompt': prompt,
            'negative_prompt': '',   # required by KIE even when empty
            'aspect_ratio': ratio,
            'duration': int(safe_dur),  # KIE expects integer, not string
        }
        if mode and is_kling3:
            payload_input['mode'] = mode
        if image_url:
            payload_input['image_url'] = image_url

    headers = {'Authorization': f'Bearer {KIE_API_KEY}', 'Content-Type': 'application/json'}

    video_url, err = _kie_submit_and_poll(job_id, kie_model, payload_input, socket_id, headers)

    if err and not video_url:
        log.warning(f"Video job failed [{job_id}], retrying in 5s: {err}")
        time.sleep(5)
        video_url, err = _kie_submit_and_poll(job_id, kie_model, payload_input, socket_id, headers)

    if not video_url:
        log.error(f"Video job failed [{job_id}] after retry: {err}")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': f'KIE video failed: {err}'})
        return

    # Emit IMMEDIATELY — user sees result now, no waiting for Drive
    total_cost = log_cost(job_id, 'kie', kie_model, 'video', model_name, cost_per)
    emit_to(socket_id, 'job:complete', {
        'job_id': job_id, 'url': video_url, 'cost': total_cost, 'provider': 'kie'
    })
    emit_to(socket_id, 'cost:update', {'today_total': today_total()})
    log.info(f"Video job complete [{job_id}] cost=${total_cost:.4f}")

    # Drive upload in background — never blocks the user
    if GOOGLE_SA_JSON and GOOGLE_DRIVE_FOLDER:
        date_str = datetime.date.today().strftime('%Y%m%d')
        safe_name = model_name.replace(' ', '_').lower()
        filename = f"{safe_name}_video_{date_str}_{job_id[:8]}.mp4"
        def _bg_upload(url=video_url, fname=filename):
            try:
                vid_data = requests.get(url, timeout=120).content
                upload_to_drive(vid_data, fname, 'video/mp4')
            except Exception as ex:
                log.warning(f"BG Drive upload failed [{job_id}]: {ex}")
        threading.Thread(target=_bg_upload, daemon=True).start()


# ── Video generation via Laozhang ────────────────────────────────────────────

def _parse_lz_video_url(d: dict) -> str | None:
    """Extract video URL from any Laozhang video response shape."""
    data = d.get('data')
    if isinstance(data, list) and data:
        item = data[0]
        return item.get('url') or item.get('video_url') or item.get('result', {}).get('url')
    if isinstance(data, dict):
        return data.get('url') or data.get('video_url') or data.get('result', {}).get('url')
    # Some models return URL at top level
    return d.get('url') or d.get('video_url')


def gen_video_laozhang(job_id: str, prompt: str, model: str, duration: str,
                       ratio: str, image_url: str | None, socket_id: str,
                       model_name: str = 'unknown'):
    """Generate video via Laozhang.ai.
    Handles both sync (URL in first response) and async (task ID + polling) APIs.
    Wan 2.1 and Kling models both supported.
    """
    cost_per = LAOZHANG_VIDEO_COSTS.get(model, 0.15)
    size = LAOZHANG_VIDEO_SIZE_MAP.get(ratio, '720x1280')
    headers = {'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'}

    emit_to(socket_id, 'job:progress', {'job_id': job_id, 'status': 'Submitting to Laozhang…', 'pct': 5})

    payload = {'model': model, 'prompt': prompt, 'size': size, 'duration': int(duration)}
    if image_url:
        payload['image_url'] = image_url

    # ── Step 1: Submit ────────────────────────────────────────────────────────
    try:
        with eventlet.Timeout(60):
            r = requests.post(
                'https://api.laozhang.ai/v1/video/generations',
                headers=headers, json=payload, timeout=60,
            )
        r.raise_for_status()
        d = r.json()
        log.info(f"Laozhang video submit [{job_id}] → {json.dumps(d)[:300]}")
    except eventlet.Timeout:
        log.error(f"Laozhang video submit timed out [{job_id}]")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'Laozhang submit timed out — try again'})
        return
    except Exception as e:
        log.error(f"Laozhang video submit failed [{job_id}]: {e}")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': f'Laozhang submit failed: {e}'})
        return

    # ── Step 2: Try sync URL first ────────────────────────────────────────────
    video_url = _parse_lz_video_url(d)
    if video_url:
        log.info(f"Laozhang video sync [{job_id}]: {video_url[:80]}")
    else:
        # ── Step 3: Async — extract task/generation ID and poll ───────────────
        task_id = (d.get('id')
                   or (d.get('data') or {}).get('id') if isinstance(d.get('data'), dict) else None
                   or next((item.get('id') for item in (d.get('data') or []) if isinstance(item, dict)), None))

        if not task_id:
            log.error(f"Laozhang video: no URL and no task ID [{job_id}]: {d}")
            emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'Laozhang: no task ID in response — check API key or model name'})
            return

        log.info(f"Laozhang video async [{job_id}] task_id={task_id} — polling…")
        deadline = time.time() + 600  # 10 min max
        poll_num = 0
        while time.time() < deadline:
            poll_interval = 5 if poll_num < 12 else 10
            time.sleep(poll_interval)
            poll_num += 1
            try:
                pr = requests.get(
                    f'https://api.laozhang.ai/v1/video/generations/{task_id}',
                    headers=headers, timeout=30,
                )
                pr.raise_for_status()
                pd = pr.json()
                pct = min(10 + poll_num * 4, 90)
                status = (pd.get('status') or pd.get('state') or '').lower()
                emit_to(socket_id, 'job:progress', {
                    'job_id': job_id,
                    'status': f'Generating… ({status or f"poll {poll_num}"})',
                    'pct': pct,
                })
                log.info(f"Laozhang poll [{job_id}] #{poll_num} status={status}: {json.dumps(pd)[:200]}")

                video_url = _parse_lz_video_url(pd)
                if video_url:
                    break
                if status in ('failed', 'cancelled', 'error'):
                    emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': f'Laozhang video failed: {status}'})
                    return
            except Exception as e:
                log.warning(f"Laozhang poll error [{job_id}] #{poll_num}: {e}")

        if not video_url:
            emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'Laozhang video timed out after 10 min'})
            return

    # ── Done ──────────────────────────────────────────────────────────────────
    total_cost = log_cost(job_id, 'laozhang', model, 'video', model_name, cost_per)
    emit_to(socket_id, 'job:complete', {
        'job_id': job_id, 'url': video_url, 'cost': total_cost, 'provider': 'laozhang'
    })
    emit_to(socket_id, 'cost:update', {'today_total': today_total()})
    log.info(f"Laozhang video complete [{job_id}] cost=${total_cost:.4f}")

    if GOOGLE_SA_JSON and GOOGLE_DRIVE_FOLDER:
        date_str = datetime.date.today().strftime('%Y%m%d')
        safe_name = model_name.replace(' ', '_').lower()
        fname = f"{safe_name}_video_{date_str}_{job_id[:8]}.mp4"
        def _bg_upload(url=video_url, fn=fname):
            try:
                vid_data = requests.get(url, timeout=120).content
                upload_to_drive(vid_data, fn, 'video/mp4')
            except Exception as ex:
                log.warning(f"BG Drive upload failed [{job_id}]: {ex}")
        threading.Thread(target=_bg_upload, daemon=True).start()


# ── Image generation via KIE.ai ──────────────────────────────────────────────

# Models that need extra fields in their input (beyond just prompt + aspect_ratio)
_KIE_NB_MODELS = {'nano-banana-pro', 'nano-banana-2', 'google/nano-banana'}

def _kie_image_input(model: str, prompt: str, ratio: str) -> dict:
    """Build correct KIE input payload for each image model."""
    if model in _KIE_NB_MODELS:
        return {'prompt': prompt, 'image_input': [], 'aspect_ratio': ratio,
                'resolution': '2K', 'output_format': 'png'}
    if model == 'grok-imagine/text-to-image':
        # Grok only supports: 2:3, 3:2, 1:1, 16:9, 9:16
        safe_ratio = ratio if ratio in ('1:1', '9:16', '16:9', '3:4', '4:3') else '1:1'
        gr = {'9:16': '2:3', '16:9': '3:2', '3:4': '2:3', '4:3': '3:2', '1:1': '1:1'}.get(ratio, '1:1')
        return {'prompt': prompt, 'aspect_ratio': gr, 'nsfw_checker': False, 'enable_pro': True}
    if model in ('google/imagen4', 'google/imagen4-ultra'):
        return {'prompt': prompt, 'aspect_ratio': ratio, 'negative_prompt': '', 'seed': ''}
    # Default: simple prompt + aspect_ratio (works for gpt-image-2-text-to-image etc.)
    return {'prompt': prompt, 'aspect_ratio': ratio}


def _kie_extract_image_url(result: dict) -> str | None:
    """Extract image URL from KIE result — handles multiple response shapes."""
    return (result.get('imageUrls', [None])[0]
            or result.get('resultUrls', [None])[0] if result.get('resultUrls') else None
            or next((v['url'] for v in result.get('images', []) if v.get('url')), None)
            or result.get('url'))


def gen_image_kie(job_id: str, prompt: str, ratio: str, model_name: str,
                  socket_id: str, model: str = 'nano-banana-pro'):
    """Generate image via KIE.ai, emit result on completion."""
    cost_per = KIE_IMAGE_COSTS.get(model, 0.040)
    headers = {'Authorization': f'Bearer {KIE_API_KEY}', 'Content-Type': 'application/json'}
    inp = _kie_image_input(model, prompt, ratio)

    emit_to(socket_id, 'job:progress', {'job_id': job_id, 'status': 'Submitting to KIE…', 'pct': 5})
    log.info(f"KIE image submit [{job_id}] model={model} input={json.dumps(inp)[:200]}")

    try:
        with eventlet.Timeout(30):
            r = requests.post(
                f'{KIE_BASE}/createTask',
                headers=headers,
                json={'model': model, 'input': inp},
                timeout=30,
            )
        d = r.json()
        log.info(f"KIE image createTask [{job_id}]: {json.dumps(d)[:200]}")
        if d.get('code') != 200:
            raise Exception(f"KIE error {d.get('code')}: {d.get('msg', 'unknown')}")
        task_id = d['data']['taskId']
        log.info(f"KIE image task created [{job_id}]: {task_id}")
    except eventlet.Timeout:
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'KIE submit timed out — try again'})
        return
    except Exception as e:
        log.error(f"KIE image submit failed [{job_id}]: {e}")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': f'KIE submit failed: {e}'})
        return

    deadline = time.time() + 300  # 5 min max
    poll_num = 0
    while time.time() < deadline:
        time.sleep(3)
        poll_num += 1
        try:
            pr = requests.get(f'{KIE_BASE}/recordInfo?taskId={task_id}', headers=headers, timeout=15)
            pd = pr.json()
            raw = pd.get('data', {})
            if isinstance(raw.get('resultJson'), str):
                try: raw['result'] = json.loads(raw['resultJson'])
                except: pass
            state = raw.get('state', '')
            pct = min(10 + poll_num * 6, 90)
            emit_to(socket_id, 'job:progress', {'job_id': job_id, 'status': f'Generating… ({state})', 'pct': pct})
            log.info(f"KIE image poll [{job_id}] #{poll_num} state={state}")
            if state == 'success':
                res = raw.get('result') or {}
                img_url = _kie_extract_image_url(res)
                if not img_url:
                    # Dump what we got so we can debug
                    log.error(f"KIE image success but no URL [{job_id}]: {json.dumps(raw)[:500]}")
                    emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'KIE returned success but no image URL — check logs'})
                    return
                total_cost = log_cost(job_id, 'kie', model, 'image', model_name, cost_per)
                emit_to(socket_id, 'job:complete', {'job_id': job_id, 'url': img_url, 'cost': total_cost, 'provider': 'kie'})
                emit_to(socket_id, 'cost:update', {'today_total': today_total()})
                log.info(f"KIE image complete [{job_id}] cost=${total_cost:.4f} url={img_url[:60]}")
                return
            elif state in ('fail', 'failed', 'error'):
                err = raw.get('failReason') or raw.get('error') or 'KIE generation failed'
                emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': f'KIE failed: {err}'})
                return
        except Exception as e:
            log.warning(f"KIE image poll error [{job_id}] #{poll_num}: {e}")

    emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'KIE image timed out after 5 min'})


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/generate/image', methods=['POST'])
def api_gen_image():
    d = request.get_json(force=True)
    t = threading.Thread(target=gen_image, daemon=True, kwargs=dict(
        job_id=d['job_id'],
        prompt=d['prompt'],
        resolution=d.get('resolution', '2K'),
        ratio=d.get('ratio', '9:16'),
        model=d.get('model', 'nano-banana-pro'),
        model_name=d.get('model_name', 'unknown'),
        ref_urls=d.get('ref_urls', []),
        socket_id=d.get('socket_id', ''),
    ))
    t.start()
    return jsonify({'ok': True, 'job_id': d['job_id']})


@app.route('/api/generate/image/batch', methods=['POST'])
def api_gen_image_batch():
    """Batch endpoint: routes to Laozhang (single n=count call) or KIE (parallel threads)."""
    d = request.get_json(force=True)
    provider = d.get('provider', 'laozhang')
    job_ids  = d['job_ids']

    if provider == 'kie':
        # KIE doesn't batch — fire one thread per image, all in parallel
        for job_id in job_ids:
            threading.Thread(target=gen_image_kie, daemon=True, kwargs=dict(
                job_id=job_id,
                prompt=d['prompt'],
                ratio=d.get('ratio', '9:16'),
                model=d.get('model', 'kolors'),
                model_name=d.get('model_name', 'unknown'),
                socket_id=d.get('socket_id', ''),
            )).start()
    else:
        # Laozhang: one API call with n=count
        threading.Thread(target=gen_image_batch, daemon=True, kwargs=dict(
            job_ids=job_ids,
            prompt=d['prompt'],
            resolution=d.get('resolution', '2K'),
            ratio=d.get('ratio', '9:16'),
            model=d.get('model', 'nano-banana-pro'),
            model_name=d.get('model_name', 'unknown'),
            ref_urls=d.get('ref_urls', []),
            socket_id=d.get('socket_id', ''),
        )).start()

    return jsonify({'ok': True, 'job_ids': job_ids})


@app.route('/api/generate/video', methods=['POST'])
def api_gen_video():
    d = request.get_json(force=True)
    provider = d.get('provider', 'kie')

    if provider == 'laozhang':
        t = threading.Thread(target=gen_video_laozhang, daemon=True, kwargs=dict(
            job_id=d['job_id'],
            prompt=d.get('prompt', ''),
            model=d.get('model', 'wan2.1-14b-720p'),
            duration=str(d.get('duration', '5')),
            ratio=d.get('ratio', '9:16'),
            image_url=d.get('image_url'),
            model_name=d.get('model_name', 'unknown'),
            socket_id=d.get('socket_id', ''),
        ))
    else:
        t = threading.Thread(target=gen_video, daemon=True, kwargs=dict(
            job_id=d['job_id'],
            prompt=d.get('prompt', ''),
            model=d.get('model', 'kling/v2-1-pro'),
            duration=str(d.get('duration', '5')),
            ratio=d.get('ratio', '9:16'),
            image_url=d.get('image_url'),
            mode=d.get('mode'),
            model_name=d.get('model_name', 'unknown'),
            socket_id=d.get('socket_id', ''),
            mc_input_urls=d.get('mc_input_urls'),
            mc_video_urls=d.get('mc_video_urls'),
            mc_orientation=d.get('mc_orientation'),
        ))
    t.start()
    return jsonify({'ok': True, 'job_id': d['job_id']})


@app.route('/api/costs/summary')
def api_costs_summary():
    with get_db() as db:
        today      = datetime.date.today().isoformat()
        week_start = (datetime.date.today() - datetime.timedelta(days=datetime.date.today().weekday())).isoformat()
        month_start = datetime.date.today().replace(day=1).isoformat()

        def period_sum(start, end=None):
            if end:
                row = db.execute(
                    "SELECT COALESCE(SUM(total_cost),0) t FROM cost_log WHERE date(generated_at) BETWEEN ? AND ?",
                    (start, end)).fetchone()
            else:
                row = db.execute(
                    "SELECT COALESCE(SUM(total_cost),0) t FROM cost_log WHERE date(generated_at) >= ?",
                    (start,)).fetchone()
            return round(float(row['t']), 4)

        td = period_sum(today, today)
        wk = period_sum(week_start)
        mo = period_sum(month_start)

        days_so_far = datetime.date.today().day
        projection = round(mo / days_so_far * 30, 2) if days_so_far > 0 else 0

        # Per-model breakdown this week
        rows = db.execute(
            "SELECT model_name, content_type, COUNT(*) cnt, COALESCE(SUM(total_cost),0) cost "
            "FROM cost_log WHERE date(generated_at) >= ? GROUP BY model_name, content_type",
            (week_start,)).fetchall()
        by_model = {}
        for r in rows:
            mn = r['model_name']
            if mn not in by_model:
                by_model[mn] = {'images': 0, 'image_cost': 0.0, 'videos': 0, 'video_cost': 0.0}
            if r['content_type'] == 'image':
                by_model[mn]['images']     += r['cnt']
                by_model[mn]['image_cost'] += round(float(r['cost']), 4)
            else:
                by_model[mn]['videos']     += r['cnt']
                by_model[mn]['video_cost'] += round(float(r['cost']), 4)

        wk_img = db.execute(
            "SELECT COALESCE(SUM(total_cost),0) t, COUNT(*) c FROM cost_log "
            "WHERE date(generated_at) >= ? AND content_type='image'", (week_start,)).fetchone()
        wk_vid = db.execute(
            "SELECT COALESCE(SUM(total_cost),0) t, COUNT(*) c FROM cost_log "
            "WHERE date(generated_at) >= ? AND content_type='video'", (week_start,)).fetchone()

    return jsonify({
        'today': td,
        'week':  wk,
        'month': mo,
        'projection_monthly': projection,
        'by_model': by_model,
        'week_images': {'count': wk_img['c'], 'cost': round(float(wk_img['t']), 4)},
        'week_videos': {'count': wk_vid['c'], 'cost': round(float(wk_vid['t']), 4)},
    })


@app.route('/api/costs/log')
def api_costs_log():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM cost_log ORDER BY generated_at DESC LIMIT 200"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/test/laozhang')
def api_test_laozhang():
    model = request.args.get('model', 'seedream/4.5-text-to-image')
    try:
        r = requests.post(
            'https://api.laozhang.ai/v1/images/generations',
            headers={'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': model, 'prompt': 'a beautiful woman, photography', 'n': 1, 'size': '512x910', 'quality': 'hd'},
            timeout=30,
        )
        try:
            body = r.json()
        except Exception:
            body = r.text[:500]
        ok = r.status_code in (200, 201)
        return jsonify({'status': 'ok' if ok else 'error', 'provider': 'laozhang',
                        'model': model, 'http': r.status_code, 'response': body})
    except Exception as e:
        return jsonify({'status': 'error', 'provider': 'laozhang', 'error': str(e)}), 502


@app.route('/api/test/kie')
def api_test_kie():
    try:
        r = requests.get(
            f'{KIE_BASE}/recordInfo?taskId=healthcheck',
            headers={'Authorization': f'Bearer {KIE_API_KEY}'},
            timeout=10,
        )
        try:
            body = r.json()
        except Exception:
            body = r.text[:500]
        ok = r.status_code in (200, 400, 404, 422)
        return jsonify({'status': 'ok' if ok else 'error', 'provider': 'kie',
                        'http': r.status_code, 'response': body})
    except Exception as e:
        return jsonify({'status': 'error', 'provider': 'kie', 'error': str(e)}), 502


@app.route('/api/debug/image')
def api_debug_image():
    """Full diagnostic — tries a real generation and returns raw API response. Do NOT use in production load."""
    model = request.args.get('model', 'seedream/4.5-text-to-image')
    prompt = request.args.get('prompt', 'a beautiful woman smiling, studio photography')
    key_hint = (LAOZHANG_API_KEY[:8] + '…') if LAOZHANG_API_KEY else 'NOT SET'
    try:
        payload = _lz_payload(model, prompt, '9:16', '2K', [], n=1)
        r = requests.post(
            'https://api.laozhang.ai/v1/images/generations',
            headers={'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=60,
        )
        try:
            body = r.json()
        except Exception:
            body = r.text[:2000]
        # If b64_json returned, summarize instead of dumping megabytes
        if isinstance(body, dict) and body.get('data'):
            for item in body['data']:
                if item.get('b64_json'):
                    item['b64_json'] = f'<{len(item["b64_json"])} chars of base64>'
        return jsonify({
            'key_hint': key_hint,
            'model': model,
            'http': r.status_code,
            'payload_sent': payload,
            'response': body,
        })
    except Exception as e:
        return jsonify({'key_hint': key_hint, 'model': model, 'error': str(e)}), 502


@app.route('/api/debug/kie-image')
def api_debug_kie_image():
    """Full KIE image generation test — submits real task and polls to completion."""
    model = request.args.get('model', 'nano-banana-pro')
    prompt = request.args.get('prompt', 'a beautiful woman smiling, studio photography')
    key_hint = (KIE_API_KEY[:8] + '…') if KIE_API_KEY else 'NOT SET'
    headers = {'Authorization': f'Bearer {KIE_API_KEY}', 'Content-Type': 'application/json'}
    inp = _kie_image_input(model, prompt, '9:16')
    try:
        r = requests.post(f'{KIE_BASE}/createTask', headers=headers,
                          json={'model': model, 'input': inp},
                          timeout=30)
        d = r.json()
        if d.get('code') != 200:
            return jsonify({'key_hint': key_hint, 'model': model, 'submit': d})
        task_id = d['data']['taskId']
        # Poll up to 60s
        for _ in range(20):
            time.sleep(3)
            pr = requests.get(f'{KIE_BASE}/recordInfo?taskId={task_id}', headers=headers, timeout=15)
            pd = pr.json()
            if isinstance(pd['data'].get('resultJson'), str):
                try: pd['data']['result'] = json.loads(pd['data']['resultJson'])
                except: pass
            state = pd['data'].get('state')
            if state == 'success':
                res = pd['data'].get('result', {})
                urls = res.get('resultUrls') or [v['url'] for v in res.get('images', [])] or ([res['url']] if res.get('url') else [])
                return jsonify({'key_hint': key_hint, 'model': model, 'task_id': task_id, 'status': 'success', 'url': urls[0] if urls else None})
            if state == 'fail':
                return jsonify({'key_hint': key_hint, 'model': model, 'task_id': task_id, 'status': 'fail', 'data': pd['data']})
        return jsonify({'key_hint': key_hint, 'model': model, 'task_id': task_id, 'status': 'still_polling — check manually'})
    except Exception as e:
        return jsonify({'key_hint': key_hint, 'model': model, 'error': str(e)}), 502


@app.route('/api/debug/kie-video')
def api_debug_kie_video():
    """Dry-run KIE video createTask — shows EXACTLY what we send and what KIE replies.
    Does NOT poll to completion (video takes too long). Just verifies the submit succeeds.
    Usage: /api/debug/kie-video?model=kling/v2-1-standard&duration=5&ratio=9:16
    """
    model    = request.args.get('model', 'kling/v2-1-standard')
    duration = int(request.args.get('duration', '5'))
    ratio    = request.args.get('ratio', '9:16')
    mode     = request.args.get('mode', 'std')
    key_hint = (KIE_API_KEY[:8] + '…') if KIE_API_KEY else 'NOT SET'
    headers  = {'Authorization': f'Bearer {KIE_API_KEY}', 'Content-Type': 'application/json'}
    payload_input = {
        'prompt': 'beautiful woman walking on beach, cinematic',
        'negative_prompt': '',
        'aspect_ratio': ratio,
        'duration': duration,
        'mode': mode,
    }
    body = {'model': model, 'input': payload_input}
    try:
        r = requests.post(f'{KIE_BASE}/createTask', headers=headers, json=body, timeout=30)
        try:
            resp = r.json()
        except Exception:
            resp = r.text[:500]
        return jsonify({'key_hint': key_hint, 'model': model,
                        'http': r.status_code, 'payload_sent': body, 'response': resp})
    except Exception as e:
        return jsonify({'key_hint': key_hint, 'model': model, 'error': str(e)}), 502


@app.route('/api/test/all')
def api_test_all():
    results = {}
    if not LAOZHANG_API_KEY:
        results['laozhang'] = 'no_key'
    if not KIE_API_KEY:
        results['kie'] = 'no_key'

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        if LAOZHANG_API_KEY:
            futures['laozhang'] = ex.submit(lambda: requests.post(
                'https://api.laozhang.ai/v1/images/generations',
                headers={'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'},
                json={'model': 'nano-banana-pro', 'prompt': 'test', 'n': 1, 'size': '512x910', 'quality': 'hd'},
                timeout=15))
        if KIE_API_KEY:
            futures['kie'] = ex.submit(lambda: requests.get(
                f'{KIE_BASE}/recordInfo?taskId=healthcheck',
                headers={'Authorization': f'Bearer {KIE_API_KEY}'}, timeout=10))

    if 'laozhang' in futures:
        try:
            lr = futures['laozhang'].result()
            results['laozhang'] = 'ok' if lr.status_code in (200, 201, 400, 422, 503) else 'error'
        except Exception:
            results['laozhang'] = 'error'
    if 'kie' in futures:
        try:
            kr = futures['kie'].result()
            results['kie'] = 'ok' if kr.status_code in (200, 400, 404, 422) else 'error'
        except Exception:
            results['kie'] = 'error'

    results['all_ok'] = results.get('laozhang') == 'ok' and results.get('kie') == 'ok'
    return jsonify(results)


# ── Socket.io events ──────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    log.info(f"Socket.io client connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    log.info(f"Socket.io client disconnected: {request.sid}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    log.info(f"Starting Cuzo Content Factory on port {port}")
    socketio.run(app, host='0.0.0.0', port=port)
