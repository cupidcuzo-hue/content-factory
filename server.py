"""
Cuzo Content Factory — Python Backend
  Images → Laozhang.ai (nano-banana-pro)
  Videos → KIE.ai (Kling 2.6 / 3.0 / v2-1)
  Cost tracking → SQLite
  Real-time progress → Socket.io
  Storage → Google Drive
"""

import os, sys, json, time, io, threading, sqlite3, datetime, logging
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO

# ── Startup env checks ────────────────────────────────────────────────────────

LAOZHANG_API_KEY = os.environ.get('LAOZHANG_API_KEY')
KIE_API_KEY      = os.environ.get('KIE_API_KEY')

if not LAOZHANG_API_KEY:
    sys.exit("❌  LAOZHANG_API_KEY env var is missing. Set it in Railway Variables.")
if not KIE_API_KEY:
    sys.exit("❌  KIE_API_KEY env var is missing. Set it in Railway Variables.")

GOOGLE_SA_JSON       = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON', '')
GOOGLE_DRIVE_FOLDER  = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', '')
DB_PATH              = os.environ.get('DB_PATH', 'data/content_factory.db')

# ── Flask / SocketIO ──────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cuzo-cf-secret-2025')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
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
    log.info(f"Database ready at {DB_PATH}")

init_db()

# ── Cost constants ────────────────────────────────────────────────────────────

LAOZHANG_COSTS = {'1K': 0.0125, '2K': 0.025, '4K': 0.050}

VIDEO_COSTS = {
    'kling/v2-1-standard':      0.125,
    'kling/v2-1-pro':           0.250,
    'kling/v2-1-master':        0.800,
    'kling-3.0/video':          0.350,
    'kling-2.6/text-to-video':  0.250,
    'kling-3.0/motion-control': 0.500,
    'kling-2.6/motion-control': 0.350,
}

IMG_SIZE_MAP = {
    ('9:16', '2K'): '1024x1820', ('9:16', '4K'): '2048x3640', ('9:16', '1K'): '512x910',
    ('1:1',  '2K'): '1024x1024', ('1:1',  '4K'): '2048x2048', ('1:1',  '1K'): '512x512',
    ('4:3',  '2K'): '1024x768',  ('4:3',  '4K'): '2048x1536', ('4:3',  '1K'): '512x384',
    ('3:4',  '2K'): '768x1024',  ('3:4',  '4K'): '1536x2048', ('3:4',  '1K'): '384x512',
    ('16:9', '2K'): '1820x1024', ('16:9', '4K'): '3640x2048', ('16:9', '1K'): '910x512',
}

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

def gen_image(job_id: str, prompt: str, resolution: str, ratio: str,
              model_name: str, ref_urls: list, socket_id: str):
    """Generate image via Laozhang NanoBanana Pro, upload to Drive, log cost."""
    size = IMG_SIZE_MAP.get((ratio, resolution), '1024x1820')
    cost_per = LAOZHANG_COSTS.get(resolution, 0.025)

    emit_to(socket_id, 'job:progress', {'job_id': job_id, 'status': 'Generating image via Laozhang…', 'pct': 10})

    payload = {
        'model': 'nano-banana-pro',
        'prompt': prompt,
        'n': 1,
        'size': size,
        'quality': 'hd',
    }
    # Include reference images if provided (API may support image_input)
    if ref_urls:
        payload['image_input'] = ref_urls

    try:
        r = requests.post(
            'https://api.laozhang.ai/v1/images/generations',
            headers={'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        img_url = data['data'][0]['url']
        log.info(f"Laozhang image ready [{job_id}]: {img_url[:60]}…")
    except Exception as e:
        log.error(f"Laozhang image gen failed [{job_id}]: {e}")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'Image generation failed — Laozhang API error'})
        return  # No fallback, no cost log

    emit_to(socket_id, 'job:progress', {'job_id': job_id, 'status': 'Downloading & uploading to Drive…', 'pct': 75})

    # Download
    try:
        img_data = requests.get(img_url, timeout=60).content
    except Exception as e:
        log.error(f"Image download failed [{job_id}]: {e}")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'Image generation failed — download error'})
        return

    # Upload to Drive
    date_str = datetime.date.today().strftime('%Y%m%d')
    safe_name = model_name.replace(' ', '_').lower()
    filename = f"{safe_name}_image_{date_str}_{job_id[:8]}.jpg"
    drive_url = upload_to_drive(img_data, filename, 'image/jpeg')
    final_url = drive_url or img_url  # Fall back to Laozhang URL if Drive not configured

    # Log cost & emit
    total_cost = log_cost(job_id, 'laozhang', 'nano-banana-pro', 'image', model_name, cost_per)

    emit_to(socket_id, 'job:complete', {
        'job_id': job_id, 'url': final_url, 'cost': total_cost, 'provider': 'laozhang'
    })
    emit_to(socket_id, 'cost:update', {'today_total': today_total()})
    log.info(f"Image job complete [{job_id}] cost=${total_cost:.4f}")


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
        time.sleep(5)
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
            'aspect_ratio': ratio,
            'duration': str(safe_dur),
        }
        if mode and is_kling3:
            payload_input['mode'] = mode
        if image_url:
            payload_input['image_urls'] = [image_url]

    headers = {'Authorization': f'Bearer {KIE_API_KEY}', 'Content-Type': 'application/json'}

    video_url, err = _kie_submit_and_poll(job_id, kie_model, payload_input, socket_id, headers)

    if err and not video_url:
        # Retry once after 30 seconds
        log.warning(f"Video job failed [{job_id}], retrying in 30s: {err}")
        time.sleep(30)
        video_url, err = _kie_submit_and_poll(job_id, kie_model, payload_input, socket_id, headers)

    if not video_url:
        log.error(f"Video job failed [{job_id}] after retry: {err}")
        emit_to(socket_id, 'job:failed', {'job_id': job_id, 'error': 'Video generation failed — KIE API error'})
        return

    emit_to(socket_id, 'job:progress', {'job_id': job_id, 'status': 'Uploading to Drive…', 'pct': 92})

    # Download
    vid_data = None
    try:
        vid_data = requests.get(video_url, timeout=120).content
    except Exception as e:
        log.error(f"Video download failed [{job_id}]: {e}")

    # Upload to Drive
    date_str = datetime.date.today().strftime('%Y%m%d')
    safe_name = model_name.replace(' ', '_').lower()
    filename = f"{safe_name}_video_{date_str}_{job_id[:8]}.mp4"
    drive_url = upload_to_drive(vid_data, filename, 'video/mp4') if vid_data else ''
    final_url = drive_url or video_url

    total_cost = log_cost(job_id, 'kie', kie_model, 'video', model_name, cost_per)

    emit_to(socket_id, 'job:complete', {
        'job_id': job_id, 'url': final_url, 'cost': total_cost, 'provider': 'kie'
    })
    emit_to(socket_id, 'cost:update', {'today_total': today_total()})
    log.info(f"Video job complete [{job_id}] cost=${total_cost:.4f}")


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
        model_name=d.get('model_name', 'unknown'),
        ref_urls=d.get('ref_urls', []),
        socket_id=d.get('socket_id', ''),
    ))
    t.start()
    return jsonify({'ok': True, 'job_id': d['job_id']})


@app.route('/api/generate/video', methods=['POST'])
def api_gen_video():
    d = request.get_json(force=True)
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
    try:
        r = requests.post(
            'https://api.laozhang.ai/v1/images/generations',
            headers={'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': 'nano-banana-pro', 'prompt': 'connectivity test', 'n': 1, 'size': '64x64'},
            timeout=15,
        )
        # 400/422 = API reachable but bad request — still means creds work
        ok = r.status_code in (200, 201, 400, 422)
        return jsonify({'status': 'ok' if ok else 'error', 'provider': 'laozhang',
                        'model': 'nano-banana-pro', 'http': r.status_code})
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
        ok = r.status_code in (200, 400, 404, 422)
        return jsonify({'status': 'ok' if ok else 'error', 'provider': 'kie',
                        'model': 'kling-2.1', 'http': r.status_code})
    except Exception as e:
        return jsonify({'status': 'error', 'provider': 'kie', 'error': str(e)}), 502


@app.route('/api/test/all')
def api_test_all():
    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        lf = ex.submit(lambda: requests.post(
            'https://api.laozhang.ai/v1/images/generations',
            headers={'Authorization': f'Bearer {LAOZHANG_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': 'nano-banana-pro', 'prompt': 'test', 'n': 1, 'size': '64x64'},
            timeout=15))
        kf = ex.submit(lambda: requests.get(
            f'{KIE_BASE}/recordInfo?taskId=healthcheck',
            headers={'Authorization': f'Bearer {KIE_API_KEY}'}, timeout=10))
    try:
        lr = lf.result(); results['laozhang'] = 'ok' if lr.status_code in (200, 201, 400, 422) else 'error'
    except Exception:
        results['laozhang'] = 'error'
    try:
        kr = kf.result(); results['kie'] = 'ok' if kr.status_code in (200, 400, 404, 422) else 'error'
    except Exception:
        results['kie'] = 'error'
    results['all_ok'] = results['laozhang'] == 'ok' and results['kie'] == 'ok'
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
