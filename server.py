"""
HairLens AI - WebSocket Server v2
MediaPipe Face Mesh (468 landmarks) → hairline extraction → WebSocket stream

Usage:
    pip install mediapipe websockets opencv-python numpy
    python server.py
"""

import asyncio
import sys
import traceback

# Windows PowerShell에서 print/log이 버퍼링돼서 안 보이는 문제 방지 — 라인 단위 flush 강제
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# stdout 즉시 flush — 부팅 중 죽을 때 메시지 잘리는 것 방지
# (stderr.flush()는 mediapipe가 fd를 건드려서 Windows에서 OSError 발생 가능 → 호출 금지)
def _p(msg):
    try:
        print(msg, flush=True)
    except OSError:
        pass

_p("[boot] importing modules...")
try:
    import websockets
    import cv2
    import numpy as np
    import base64
    import json
    import os
    import time
    import logging
    import urllib.request
    from collections import deque

    import mediapipe as mp
    _p(f"[boot] mediapipe {mp.__version__} loaded")
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    _p("[boot] mediapipe.tasks loaded")

    # Claude API (선택) — 키 없거나 SDK 미설치면 비활성화하고 클라 폴백 사용
    try:
        import anthropic
        _p(f"[boot] anthropic SDK {getattr(anthropic, '__version__', '?')} loaded")
    except ImportError:
        anthropic = None
        _p("[boot] anthropic SDK 미설치 → AI 리포트 비활성화 (pip install anthropic 으로 활성화)")
except Exception as e:
    _p(f"[boot] IMPORT FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,   # 기본 stderr → stdout으로 (Windows PowerShell에서 더 잘 보임)
)
log = logging.getLogger("HairLens")

# ── Config ─────────────────────────────────────────────────────────────────
WS_HOST      = "0.0.0.0"
WS_PORT      = 8765
CAMERA_INDEX = 0
TARGET_FPS   = 60
JPEG_QUALITY = 75

# ── MediaPipe FaceMesh ──────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.5,
)

# ── MediaPipe Selfie Multiclass Segmenter ──────────────────────────────────
# 카테고리: 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others
SEG_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
SEG_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "selfie_multiclass_256x256.tflite")

CAT_HAIR      = 1
CAT_BODY_SKIN = 2
CAT_FACE_SKIN = 3


def ensure_seg_model():
    if os.path.exists(SEG_MODEL_PATH):
        _p(f"[boot] model exists: {SEG_MODEL_PATH} ({os.path.getsize(SEG_MODEL_PATH)//1024} KB)")
        return
    _p(f"[boot] downloading segmentation model: {SEG_MODEL_URL}")
    try:
        urllib.request.urlretrieve(SEG_MODEL_URL, SEG_MODEL_PATH)
        _p(f"[boot] saved: {SEG_MODEL_PATH} ({os.path.getsize(SEG_MODEL_PATH)//1024} KB)")
    except Exception as e:
        _p(f"[boot] model download failed: {type(e).__name__}: {e}")
        raise


try:
    ensure_seg_model()
except Exception:
    traceback.print_exc()
    sys.exit(1)

_p("[boot] creating segmenter...")
segmenter = None
try:
    # Windows 경로 이슈 회피: 파일 바이트를 직접 전달
    with open(SEG_MODEL_PATH, "rb") as _mf:
        _model_bytes = _mf.read()
    segmenter = mp_vision.ImageSegmenter.create_from_options(
        mp_vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=_model_bytes),
            output_category_mask=True,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )
except Exception as e:
    _p(f"[boot] segmenter creation failed: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

if segmenter is None:
    _p("[boot] segmenter is None — aborting")
    sys.exit(1)
_p("[boot] Selfie Multiclass Segmenter loaded ✓")

# ── 모델 웜업: 더미 프레임 1회 추론 (첫 실제 프레임 지연 제거) ──────────
_p("[boot] warming up models...")
try:
    _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _dummy_rgb = cv2.cvtColor(_dummy, cv2.COLOR_BGR2RGB)
    face_mesh.process(_dummy_rgb)
    _dummy_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=_dummy_rgb)
    segmenter.segment(_dummy_mp)
    _p("[boot] models warmed up ✓")
except Exception as e:
    _p(f"[boot] warmup warning: {e}")

# ── 자동 캡처: 얼굴 가이드 정렬 파라미터 ────────────────────────────────────
GUIDE_CX_FRAC          = 0.50    # 가이드 중심 x (frame 너비 비율)
GUIDE_CY_FRAC          = 0.55    # 가이드 중심 y (조금 아래)
GUIDE_RY_FRAC          = 0.34    # 세로 반지름 (frame 높이 비율)
GUIDE_RX_FRAC          = 0.22    # 가로 반지름 (frame 너비 비율)
ALIGN_CENTER_TOL_FRAC  = 0.08    # 중심 허용 오차
ALIGN_SIZE_LO          = 0.78    # 얼굴/가이드 크기 비율 하한
ALIGN_SIZE_HI          = 1.25    # 상한
ALIGN_EB_TOL_DEG       = 7.0     # 눈썹 기울기 허용
ALIGN_YAW_TOL          = 0.12    # IPD 세로/가로 비 (정면도)
HOLD_FRAMES_REQUIRED   = 24      # 1초 유지 (24fps 기준)
COOLDOWN_FRAMES        = 72      # 자동 캡처 후 3초 락

# 시간축 안정화: 최근 N개 측정값으로 중앙값/평균 계산
SMOOTH_WINDOW = 8
_smooth = {
    "forehead_mm":    deque(maxlen=SMOOTH_WINDOW),
    "forehead_ratio": deque(maxlen=SMOOTH_WINDOW),
    "m_index":        deque(maxlen=SMOOTH_WINDOW),
    "hci":            deque(maxlen=SMOOTH_WINDOW),
    "recession":      deque(maxlen=SMOOTH_WINDOW),
}

# 세그멘터 결과 캐시: 매 N프레임마다만 추론 (가장 무거운 연산 절감)
SEG_EVERY_N = 2
_seg_cache = {
    "frame_idx":   -10_000,
    "hair_mask":   None,
    "fskin_mask":  None,
    "shape":       None,    # (h, w)
}


def smooth_push(key, value):
    if value is None:
        return None
    _smooth[key].append(float(value))
    return float(np.median(_smooth[key]))

# ── 이마 상단 랜드마크 인덱스 (헤어라인에 가장 가까운 포인트) ──────────────
FOREHEAD_TOP = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]

LEFT_EYEBROW  = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
NOSE_TIP      = 1
CHIN          = 152
LEFT_IRIS_CENTER  = 468   # refine_landmarks=True 일 때만
RIGHT_IRIS_CENTER = 473

# 성인 평균 동공간 거리(IPD) — 카메라 거리에 무관한 실제 길이 환산용
AVG_IPD_MM = 63.0

# 측정 기록 저장 파일
MEASUREMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements.jsonl")
SNAPSHOTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
HISTORY_LIMIT     = 100
THUMB_W           = 160      # 사이드바 썸네일 가로 폭
THUMB_QUALITY     = 65
THUMB_RECENT_N    = 12       # 히스토리에 썸네일 base64로 포함할 최근 개수


def _pick(d: dict, *keys):
    """첫 번째 None 아닌 값 반환"""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def save_snapshot(frame: np.ndarray, ts: float) -> str:
    """주석 입혀진 프레임을 디스크에 저장. 파일명만 반환."""
    fname = f"{int(ts*1000)}.jpg"
    path = os.path.join(SNAPSHOTS_DIR, fname)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return fname


_thumb_cache = {}   # {filename: base64_str} — 스냅샷은 불변이라 단순 캐시

def make_thumb_b64(image_filename: str) -> str:
    """저장된 스냅샷에서 썸네일 base64 생성 (메모리 캐시)."""
    if not image_filename:
        return ""
    if image_filename in _thumb_cache:
        return _thumb_cache[image_filename]
    path = os.path.join(SNAPSHOTS_DIR, image_filename)
    if not os.path.exists(path):
        return ""
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > THUMB_W:
        scale = THUMB_W / w
        img = cv2.resize(img, (THUMB_W, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
    if not ok:
        return ""
    b64 = base64.b64encode(buf).decode()
    # 캐시 크기 제한 (가장 오래된 것부터 제거)
    if len(_thumb_cache) > 64:
        _thumb_cache.pop(next(iter(_thumb_cache)))
    _thumb_cache[image_filename] = b64
    return b64


def attach_image_b64(rec: dict) -> dict:
    """rec에 디스크 스냅샷의 풀-사이즈 base64를 붙여서 반환 (영속화엔 사용 안 함)."""
    fname = rec.get("image", "")
    if not fname:
        return rec
    path = os.path.join(SNAPSHOTS_DIR, fname)
    if not os.path.exists(path):
        return rec
    try:
        with open(path, "rb") as f:
            return {**rec, "image_b64": base64.b64encode(f.read()).decode()}
    except OSError:
        return rec


def save_measurement(detection: dict, frame: np.ndarray = None, ts: float = None) -> dict:
    if ts is None:
        ts = time.time()
    image_fname = ""
    if frame is not None:
        try:
            image_fname = save_snapshot(frame, ts)
        except Exception as e:
            log.warning(f"snapshot save failed: {e}")
    record = {
        "ts":              round(ts, 3),
        "image":           image_fname,
        # 측정값
        "hci":             _pick(detection, "hci_smooth",            "hci"),
        "forehead_mm":     _pick(detection, "forehead_mm_smooth",    "forehead_mm"),
        "forehead_px":     detection.get("forehead_px"),
        "forehead_ratio":  _pick(detection, "forehead_ratio_smooth", "forehead_ratio"),
        "m_index":         _pick(detection, "m_index_smooth",        "m_index"),
        "recession_ratio": _pick(detection, "recession_smooth",      "recession_ratio"),
        # 단일 분류 결과 (탈모 여부)
        "label":           detection.get("m_label"),
        "level":           detection.get("combined_level"),
    }
    with open(MEASUREMENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_history(limit: int = HISTORY_LIMIT, with_thumbs: bool = True) -> list:
    if not os.path.exists(MEASUREMENTS_FILE):
        return []
    items = []
    try:
        with open(MEASUREMENTS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError as e:
        log.warning(f"history read failed: {e}")

    # 최근 N개만 썸네일 base64 포함 (대역폭 절약)
    if with_thumbs and items:
        for it in items[-THUMB_RECENT_N:]:
            it["thumb"] = make_thumb_b64(it.get("image", ""))
    return items

# ── 탈모 색상 (3단계: 정상 / 중등도 / 심각) ──────────────────────────────
LEVEL_COLORS = [
    (0, 255, 120),   # 0 정상
    (0, 140, 255),   # 1 중등도
    (0,  60, 255),   # 2 심각
]

# ── 색상 ────────────────────────────────────────────────────────────────────
CLR_HAIRLINE = (0, 255, 200)
CLR_FOREHEAD = (0, 180, 255)
CLR_GOOD     = (0, 255, 120)
CLR_BAD      = (0, 80, 255)
CLR_HAIRLINE_GLOW = (0, 200, 160)


def classify_baldness(m_index, forehead_ratio, hci):
    """
    탈모 분류 — 3가지 독립 신호 중 가장 심한 것으로 판단.

    신호 1) 헤어라인 진폭 (m_index)
        - 헤어라인이 일자에 가까울수록 정상 (격차 작음)
        - 격차가 클수록 탈모 신호 (M자, 비대칭 후퇴 등)

    신호 2) 이마 비율 (forehead_ratio)
        - rule of thirds로 정상 ≈ 0.50
        - 클수록 이마가 넓음 = 전반적 후퇴

    신호 3) 두피 노출 (HCI)
        - HCI 낮을수록 머리카락 영역에 두피가 많이 노출됨
        - 광범위 탈모 / 대머리 잡기 위함

    각 신호를 0~3으로 분류 후 max() → 최종 레벨.
    """
    levels = []

    # 3단계 분류: 정상 / 중등도 / 심각
    # (이전 정상+경증 → 정상으로 통합, 심각 범위 살짝 확장)

    if m_index is not None:
        if   m_index < 0.42: levels.append(0)   # 정상 (이전 정상 + 경증)
        elif m_index < 0.50: levels.append(1)   # 중등도
        else:                levels.append(2)   # 심각 (≥0.50, 이전 0.52)

    if forehead_ratio is not None:
        if   forehead_ratio < 0.77: levels.append(0)
        elif forehead_ratio < 0.88: levels.append(1)
        else:                       levels.append(2)   # ≥0.88, 이전 0.91

    if hci is not None:
        if   hci > 0.52: levels.append(0)
        elif hci > 0.35: levels.append(1)
        else:            levels.append(2)   # ≤0.35, 이전 0.32

    if not levels:
        return "—", -1

    lvl = max(levels)
    return ["정상", "중등도", "심각"][lvl], lvl


def get_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def angle_of_line(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


def process_frame(frame: np.ndarray, frame_idx: int, live_only: bool = True):
    """
    live_only=True (기본, 라이브 스트림용):
        face mesh + 정렬 가이드만 — 빠름 (~10ms)
        세그멘터/HCI/헤어라인/M자/forehead_mm 모두 None
    live_only=False (캡처 순간만):
        세그멘터 + 모든 메트릭 + 풀 시각화
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    detections = []

    hair_mask = None
    fskin_mask = None

    # ── 풀 모드일 때만 세그멘터 + 머리카락 오버레이 ─────────────────
    if not live_only:
        cache_ok = (
            _seg_cache["hair_mask"] is not None
            and _seg_cache["shape"] == (h, w)
            and (frame_idx - _seg_cache["frame_idx"]) < SEG_EVERY_N
        )
        if cache_ok:
            hair_mask  = _seg_cache["hair_mask"]
            fskin_mask = _seg_cache["fskin_mask"]
        else:
            mp_img     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            seg_result = segmenter.segment(mp_img)
            cat_mask   = seg_result.category_mask.numpy_view()
            hair_mask  = (cat_mask == CAT_HAIR).astype(np.uint8) * 255
            fskin_mask = (cat_mask == CAT_FACE_SKIN).astype(np.uint8) * 255
            _seg_cache["hair_mask"]  = hair_mask
            _seg_cache["fskin_mask"] = fskin_mask
            _seg_cache["frame_idx"]  = frame_idx
            _seg_cache["shape"]      = (h, w)

        hair_idx = hair_mask > 0
        if hair_idx.any():
            purple = np.array([200, 60, 220], dtype=np.uint16)
            frame_h = frame[hair_idx].astype(np.uint16)
            frame[hair_idx] = ((frame_h * 205 + purple * 51) >> 8).astype(np.uint8)

    # ── 얼굴 정렬 가이드 (타원) ─────────────────────────────────────
    g_cx = int(w * GUIDE_CX_FRAC)
    g_cy = int(h * GUIDE_CY_FRAC)
    g_rx = int(w * GUIDE_RX_FRAC)
    g_ry = int(h * GUIDE_RY_FRAC)

    if not results.multi_face_landmarks:
        cv2.ellipse(frame, (g_cx, g_cy), (g_rx, g_ry),
                    0, 0, 360, (60, 100, 130), 2, cv2.LINE_AA)
        return frame, detections

    for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
        lms = face_landmarks.landmark

        # ── 얼굴 bbox ────────────────────────────────────────────────────
        xs = [int(lm.x * w) for lm in lms]
        ys = [int(lm.y * h) for lm in lms]
        x1 = max(min(xs) - 10, 0)
        y1 = max(min(ys) - 10, 0)
        x2 = min(max(xs) + 10, w)
        y2 = min(max(ys) + 10, h)

        # ── 기준점 ───────────────────────────────────────────────────────
        nose_px        = get_px(lms[NOSE_TIP], w, h)
        forehead_center = get_px(lms[10], w, h)
        face_height    = abs(nose_px[1] - forehead_center[1])

        # ── 이마 랜드마크 픽셀 (가이드용으로만 보존) ───────────────────
        forehead_pts = [get_px(lms[i], w, h) for i in FOREHEAD_TOP]

        # ── 눈썹 ─────────────────────────────────────────────────────
        lb_pts   = [get_px(lms[i], w, h) for i in LEFT_EYEBROW]
        rb_pts   = [get_px(lms[i], w, h) for i in RIGHT_EYEBROW]
        lb_ctr   = (int(np.mean([p[0] for p in lb_pts])),
                    int(np.mean([p[1] for p in lb_pts])))
        rb_ctr   = (int(np.mean([p[0] for p in rb_pts])),
                    int(np.mean([p[1] for p in rb_pts])))
        eb_angle = angle_of_line(lb_ctr, rb_ctr)
        is_level = abs(eb_angle) < 5.0
        clr_eb   = CLR_GOOD if is_level else CLR_BAD
        eb_top_y = min(p[1] for p in lb_pts + rb_pts)  # 눈썹 중 가장 위쪽

        # ── 헤어라인 = 컬럼별 피부↔머리 경계 (M자 보존) ───────────────
        # 이마 ROI: 측두부 후퇴까지 다 잡히도록 위쪽을 충분히 확장
        face_w = x2 - x1
        fr_x1  = max(x1 + int(face_w * 0.05), 0)
        fr_x2  = min(x2 - int(face_w * 0.05), w)
        fr_y1  = max(y1 - int(face_height * 2.5), 0)   # ← 위로 더 확장
        fr_y2  = max(eb_top_y - 4, fr_y1 + 1)

        final_hairline = []
        hci = None  # Hair Coverage Index (0~1, 클수록 머리카락 풍성)

        # ── 풀 모드일 때만 헤어라인/HCI 계산 ─────────────────────────
        if not live_only and fskin_mask is not None and fr_y2 > fr_y1 and fr_x2 > fr_x1:
            skin_roi = fskin_mask[fr_y1:fr_y2, fr_x1:fr_x2]
            if skin_roi.size > 0:
                rh, rw = skin_roi.shape
                for col in range(0, rw, 2):
                    skin_ys = np.where(skin_roi[:, col] > 0)[0]
                    if len(skin_ys) > 0:
                        final_hairline.append(
                            (fr_x1 + col, fr_y1 + int(skin_ys[0]))
                        )
                if len(final_hairline) > 4:
                    ys_arr = np.array([p[1] for p in final_hairline], dtype=np.int32)
                    med_y  = int(np.median(ys_arr))
                    h_ref  = max(face_height, 30)
                    final_hairline = [
                        p for p in final_hairline
                        if abs(p[1] - med_y) < h_ref * 2.0
                    ]

            hd_y2 = max(int(eb_top_y - face_height * 0.4), 0)
            hd_y1 = max(int(eb_top_y - face_height * 2.5), 0)
            if hd_y2 > hd_y1 and fr_x2 > fr_x1:
                zone_hair = hair_mask[hd_y1:hd_y2, fr_x1:fr_x2]
                zone_skin = fskin_mask[hd_y1:hd_y2, fr_x1:fr_x2]
                hair_n = int(np.count_nonzero(zone_hair))
                skin_n = int(np.count_nonzero(zone_skin))
                denom  = hair_n + skin_n
                if denom > 200:
                    hci = round(hair_n / denom, 3)

        # ── 그리기 ─────────────────────────────────────────────────────
        # 얼굴 외곽선 (FOREHEAD_TOP은 얼굴 oval을 한 바퀴 그리는 인덱스 배열)
        oval_pts = np.array(forehead_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [oval_pts], False, (120, 200, 255), 1, cv2.LINE_AA)

        # 눈썹 수평선
        cv2.line(frame, lb_ctr, rb_ctr, clr_eb, 1, cv2.LINE_AA)
        cv2.circle(frame, lb_ctr, 3, clr_eb, -1, cv2.LINE_AA)
        cv2.circle(frame, rb_ctr, 3, clr_eb, -1, cv2.LINE_AA)

        # 헤어라인 선 (캡처 모드일 때만)
        if len(final_hairline) > 1:
            pts_arr = np.array(final_hairline, dtype=np.int32).reshape(-1, 1, 2)
            glow = frame.copy()
            cv2.polylines(glow, [pts_arr], False, CLR_HAIRLINE_GLOW, 8, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [pts_arr], False, CLR_HAIRLINE, 2, cv2.LINE_AA)

        # ── 탈모 심각도: 헤어라인 ↔ 눈썹 거리 / 눈썹 ↔ 턱 거리 ────────
        eyebrow_y = (lb_ctr[1] + rb_ctr[1]) / 2.0
        eb_cx     = (lb_ctr[0] + rb_ctr[0]) / 2.0
        eb_half_w = abs(rb_ctr[0] - lb_ctr[0]) / 2.0 + 1.0

        # 헤어라인: ① 눈썹보다 위쪽이고 ② 두 눈썹 사이 가로 범위 안의 점만 사용
        above_eb = [
            p for p in final_hairline
            if p[1] < eyebrow_y and abs(p[0] - eb_cx) <= eb_half_w
        ]
        if above_eb:
            above_eb.sort(key=lambda p: p[0])
            n = len(above_eb)
            trim = max(0, n // 5)
            band = above_eb[trim:n - trim] if n > 2 * trim else above_eb
            hairline_y = float(np.mean([p[1] for p in band]))
        else:
            hairline_y = None

        chin_y = lms[CHIN].y * h
        forehead_px    = max(0.0, eyebrow_y - hairline_y) if hairline_y is not None else 0.0
        lower_face_px  = max(1.0, chin_y - eyebrow_y)
        ratio          = forehead_px / lower_face_px if hairline_y is not None else None

        # ── 단일 탈모 metric: 헤어라인 최대 후퇴 비율 ─────────────────────
        # = (눈썹 → 헤어라인 최상단) / (눈썹 → 턱)
        # rule of thirds로 정상은 ~0.50, 후퇴할수록 증가
        # 5th percentile 사용 → 노이즈 점 1~2개에 휘둘리지 않음
        recession_ratio = None
        if final_hairline and lower_face_px > 1:
            top_y = float(np.percentile([p[1] for p in final_hairline], 5))
            recession_ratio = round(max(0.0, eyebrow_y - top_y) / lower_face_px, 3)

        # ── M자 수치: 헤어라인 진폭 (단순 표시용, 분류엔 미사용) ──────────
        m_index = None
        if final_hairline and face_height > 0:
            ys_sorted = sorted([p[1] for p in final_hairline])
            n = len(ys_sorted)
            n_p = max(1, n // 10)
            y_high = float(np.mean(ys_sorted[:n_p]))
            y_low  = float(np.mean(ys_sorted[-n_p:]))
            m_index = round((y_low - y_high) / face_height, 3)

        # 임시 — 평활화 후 분류
        m_label = "—"
        combined_level = -1

        # ── 카메라 거리 보정: IPD(동공간 거리)로 px → mm 환산 ──────────
        forehead_mm = None
        ipd_px      = None
        if len(lms) > RIGHT_IRIS_CENTER:
            l_iris = get_px(lms[LEFT_IRIS_CENTER], w, h)
            r_iris = get_px(lms[RIGHT_IRIS_CENTER], w, h)
            ipd_px = float(np.hypot(r_iris[0] - l_iris[0], r_iris[1] - l_iris[1]))
            if ipd_px > 1.0 and hairline_y is not None:
                mm_per_px   = AVG_IPD_MM / ipd_px
                forehead_mm = round(forehead_px * mm_per_px, 1)

        # ── 가이드 정렬 체크 ────────────────────────────────────────────
        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0
        face_w_px = float(x2 - x1)

        center_off  = float(np.hypot(face_cx - g_cx, face_cy - g_cy))
        center_norm = center_off / max(min(w, h), 1)
        size_ratio  = face_w_px / max(g_rx * 2.0, 1.0)
        center_ok   = center_norm < ALIGN_CENTER_TOL_FRAC
        size_ok     = ALIGN_SIZE_LO < size_ratio < ALIGN_SIZE_HI
        level_ok    = abs(eb_angle) < ALIGN_EB_TOL_DEG
        # 정면도: 두 홍채의 |dy|/|dx| 작아야 함 (yaw가 클수록 dx↓, dy↑)
        yaw_ok = True
        if ipd_px and ipd_px > 1.0 and len(lms) > RIGHT_IRIS_CENTER:
            dy = abs(r_iris[1] - l_iris[1])
            dx = max(abs(r_iris[0] - l_iris[0]), 1)
            yaw_ok = (dy / dx) < ALIGN_YAW_TOL

        aligned = bool(center_ok and size_ok and level_ok and yaw_ok)

        # 가이드 색상: 정렬되면 hairline 청록, 아니면 회색
        guide_color = (0, 255, 200) if aligned else (90, 130, 160)
        cv2.ellipse(frame, (g_cx, g_cy), (g_rx, g_ry),
                    0, 0, 360, guide_color, 2, cv2.LINE_AA)

        # ── 시간축 평활화 ────────────────────────────────────────────────
        sm_forehead_mm    = smooth_push("forehead_mm",    forehead_mm)
        sm_forehead_ratio = smooth_push("forehead_ratio", ratio)
        sm_m_index        = smooth_push("m_index",        m_index)
        sm_hci            = smooth_push("hci",            hci)
        sm_recession      = smooth_push("recession",      recession_ratio)

        disp_ratio     = sm_forehead_ratio if sm_forehead_ratio is not None else ratio
        disp_mm        = sm_forehead_mm    if sm_forehead_mm    is not None else forehead_mm
        disp_m_index   = sm_m_index        if sm_m_index        is not None else m_index
        disp_hci       = sm_hci            if sm_hci            is not None else hci
        disp_recession = sm_recession      if sm_recession      is not None else recession_ratio

        # ── 3개 신호 중 가장 심한 것으로 분류 ────────────────────────────
        m_label, combined_level = classify_baldness(
            disp_m_index, disp_ratio, disp_hci
        )
        sev_color = LEVEL_COLORS[combined_level] if 0 <= combined_level <= 3 else (120, 120, 120)
        # 통합 점수 (시계열용): 각 신호의 정규화된 max (1.0 = 심각 임계)
        amp_n = max(0.0, (disp_m_index or 0) - 0.30) / 0.20      # 0.30~0.50 → 0~1
        fhd_n = max(0.0, (disp_ratio or 0.50) - 0.50) / 0.38     # 0.50~0.88 → 0~1
        bld_n = max(0.0, 0.72 - (disp_hci if disp_hci is not None else 1.0)) / 0.37
        combined_score = round(max(amp_n, fhd_n, bld_n), 3)

        # 헤어라인 ↔ 눈썹 가이드 라인 (캡처 모드)
        if hairline_y is not None:
            eb_mid_x = int((lb_ctr[0] + rb_ctr[0]) / 2)
            cv2.line(frame,
                     (eb_mid_x, int(eyebrow_y)),
                     (eb_mid_x, int(hairline_y)),
                     sev_color, 2, cv2.LINE_AA)

        # ── 메타데이터 ───────────────────────────────────────────────────
        hairline_y_norm = None
        if final_hairline:
            avg_y = np.mean([p[1] for p in final_hairline])
            fh = y2 - y1
            hairline_y_norm = round((avg_y - y1) / fh, 3) if fh > 0 else None

        detections.append({
            "id": face_idx,
            "bbox": [x1, y1, x2, y2],
            # 표시용 raw 값
            "forehead_mm":     forehead_mm,
            "forehead_px":     round(forehead_px, 1),
            "forehead_ratio":  round(ratio, 3) if ratio is not None else None,
            "ipd_px":          round(ipd_px, 1) if ipd_px is not None else None,
            "m_index":         m_index,
            "hci":             hci,
            # 단일 분류 결과 (탈모 여부)
            "recession_ratio": recession_ratio,
            "m_label":         m_label,           # "정상" / "경증" / "중등도" / "심각"
            "combined_score":  combined_score,    # = recession_ratio (호환성용)
            "combined_level":  combined_level,
            # 안정화 (smooth window 중앙값)
            "forehead_mm_smooth":    round(sm_forehead_mm, 1) if sm_forehead_mm is not None else None,
            "forehead_ratio_smooth": round(sm_forehead_ratio, 3) if sm_forehead_ratio is not None else None,
            "m_index_smooth":        round(sm_m_index, 3) if sm_m_index is not None else None,
            "hci_smooth":            round(sm_hci, 3) if sm_hci is not None else None,
            "recession_smooth":      round(sm_recession, 3) if sm_recession is not None else None,
            # 정렬 상태
            "aligned":      bool(aligned),
            "align_center": bool(center_ok),
            "align_size":   bool(size_ok),
            "align_level":  bool(level_ok),
            "align_yaw":    bool(yaw_ok),
            # eyebrow info (UI 호환)
            "eyebrow_angle": round(float(eb_angle), 2),
            "is_level":      bool(is_level),
        })

    # 여러 명이 잡혔을 때 가이드 중심에 가장 가까운 얼굴이 [0]번이 되도록 정렬.
    # → 화면 중앙에 정렬한 사람만 자동 캡처 대상이 됨.
    if len(detections) > 1:
        def _dist_to_guide(d):
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return (cx - g_cx) ** 2 + (cy - g_cy) ** 2
        detections.sort(key=_dist_to_guide)

    return frame, detections


# ── WebSocket 핸들러 ────────────────────────────────────────────────────────
async def stream_to_client(websocket):
    addr = websocket.remote_address
    log.info(f"Client connected: {addr}")

    # Windows CAP_DSHOW가 기본 MSMF보다 5-10배 빠르게 열림.
    # 직전 release가 완전히 풀리기 전에 새 연결이 오면 잠겨있을 수 있음 → 짧게 재시도.
    cap = None
    for attempt in range(3):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
        cap.release()
        log.warning(f"카메라 열기 재시도 {attempt+1}/3")
        await asyncio.sleep(0.4)
    if cap is None or not cap.isOpened():
        log.warning("CAP_DSHOW 실패 → 기본 백엔드 시도")
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error("카메라를 열 수 없습니다!")
        await websocket.send(json.dumps({"error": "Camera not available"}))
        return
    log.info(f"📷 camera opened (backend={int(cap.getBackendName() != '')})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    # 카메라 버퍼 최소화 → 지연 감소
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    state = {
        "last_detection":  None,
        "last_annotated":  None,
        "last_raw_frame":  None,   # 수동 캡처 시 풀 처리용 원본
        "last_frame_idx":  0,
        "aligned_frames":  0,
        "cooldown":        0,
        "auto_enabled":    True,
    }

    async def send_loop():
        frame_idx = 0
        fail_count = 0
        interval  = 1.0 / TARGET_FPS
        loop      = asyncio.get_event_loop()
        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count == 1 or fail_count % 30 == 0:
                    log.warning(f"⚠ cap.read() False (n={fail_count}) — 카메라가 다른 프로그램에 점유되었거나 끊김")
                if fail_count >= 60:
                    log.error("카메라 연속 60회 read 실패 → 재오픈 시도")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)
                    cap.open(CAMERA_INDEX, cv2.CAP_DSHOW)
                    fail_count = 0
                await asyncio.sleep(0.05)
                continue
            if fail_count > 0:
                log.info(f"✓ cap.read() recovered after {fail_count} failures")
                fail_count = 0

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()  # 캡처 시 풀 처리용으로 보관
            # 라이브: face mesh + 정렬 가이드만 (빠름)
            annotated, detections = await loop.run_in_executor(
                None, process_frame, frame.copy(), frame_idx, True
            )
            state["last_raw_frame"] = raw_frame
            state["last_frame_idx"] = frame_idx
            if detections:
                state["last_detection"] = detections[0]
                state["last_annotated"] = annotated

            # ── 자동 캡처 카운터 ────────────────────────────────────────
            # detections는 process_frame에서 가이드 중심에 가까운 순으로 정렬됨 →
            # 여러 명이 있어도 가운데 정렬된 사람만 측정됨
            auto_event = None
            if state["cooldown"] > 0:
                state["cooldown"] -= 1
                state["aligned_frames"] = 0
            elif state["auto_enabled"] and detections and detections[0].get("aligned"):
                state["aligned_frames"] += 1
                if state["aligned_frames"] >= HOLD_FRAMES_REQUIRED:
                    try:
                        # 캡처 순간만 풀 처리 (세그멘터 + 모든 메트릭 + 풀 시각화)
                        full_annotated, full_dets = await loop.run_in_executor(
                            None, process_frame, raw_frame, frame_idx, False
                        )
                        full_det = full_dets[0] if full_dets else detections[0]
                        rec = save_measurement(full_det, frame=full_annotated)
                        log.info(f"[auto] saved hci={rec.get('hci')} mm={rec.get('forehead_mm')} m={rec.get('m_index')}")
                        auto_event = {
                            "type":  "auto_saved",
                            "saved": attach_image_b64(rec),
                            "items": load_history(),
                        }
                    except Exception as e:
                        log.error(f"[auto] save failed: {e}", exc_info=True)
                    state["aligned_frames"] = 0
                    state["cooldown"] = COOLDOWN_FRAMES
            else:
                state["aligned_frames"] = 0

            # 진행도 (0~1)
            progress = (state["aligned_frames"] / HOLD_FRAMES_REQUIRED
                        if state["cooldown"] == 0 else 0.0)

            _, buf = cv2.imencode(".jpg", annotated,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            b64 = base64.b64encode(buf).decode()

            try:
                payload = json.dumps({
                    "type":     "frame",
                    "frame":    b64,
                    "detections": detections,
                    "frame_idx": frame_idx,
                    "timestamp": round(time.time(), 3),
                    "auto": {
                        "enabled":        state["auto_enabled"],
                        "aligned_frames": state["aligned_frames"],
                        "required":       HOLD_FRAMES_REQUIRED,
                        "cooldown":       state["cooldown"],
                        "progress":       round(progress, 3),
                    },
                })
            except (TypeError, ValueError) as e:
                log.error(f"frame JSON encode failed: {e} (det keys: "
                          f"{list(detections[0].keys()) if detections else None})")
                # 다음 프레임으로 넘어가서 루프 살림
                frame_idx += 1
                await asyncio.sleep(0.05)
                continue

            await websocket.send(payload)

            if auto_event is not None:
                try:
                    await websocket.send(json.dumps(auto_event))
                except Exception as e:
                    log.error(f"auto_event send failed: {e}")

            frame_idx += 1
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - t0)))

    async def recv_loop():
        async for raw in websocket:
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ctype = cmd.get("type")
            if ctype == "save":
                if state.get("last_raw_frame") is not None:
                    # 수동 저장도 풀 처리 (세그멘터 + 모든 메트릭)
                    loop = asyncio.get_event_loop()
                    full_annotated, full_dets = await loop.run_in_executor(
                        None, process_frame,
                        state["last_raw_frame"], state["last_frame_idx"], False
                    )
                    full_det = full_dets[0] if full_dets else (state.get("last_detection") or {})
                    rec = save_measurement(full_det, frame=full_annotated)
                    log.info(f"saved: hci={rec.get('hci')} mm={rec.get('forehead_mm')}")
                    await websocket.send(json.dumps({
                        "type":  "history",
                        "items": load_history(),
                        "saved": attach_image_b64(rec),
                    }))
                else:
                    await websocket.send(json.dumps({
                        "type":  "save_error",
                        "error": "측정할 얼굴이 감지되지 않습니다",
                    }))
            elif ctype == "history":
                await websocket.send(json.dumps({
                    "type":  "history",
                    "items": load_history(),
                }))
            elif ctype == "auto_toggle":
                state["auto_enabled"] = bool(cmd.get("enabled", True))
                state["aligned_frames"] = 0
                log.info(f"auto-capture: {'ON' if state['auto_enabled'] else 'OFF'}")
            elif ctype == "snapshot":
                fname = (cmd.get("file") or "").strip()
                # 경로 트래버설 방어: 파일명에 슬래시/dot-segment 금지
                if not fname or "/" in fname or "\\" in fname or ".." in fname:
                    continue
                path = os.path.join(SNAPSHOTS_DIR, fname)
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                        await websocket.send(json.dumps({
                            "type": "snapshot_full",
                            "file": fname,
                            "b64":  base64.b64encode(data).decode(),
                        }))
                    except Exception as e:
                        log.warning(f"snapshot read failed: {e}")

    try:
        await asyncio.gather(send_loop(), recv_loop())
    except websockets.exceptions.ConnectionClosedOK:
        log.info(f"Disconnected: {addr}")
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning(f"Connection error: {e}")
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
    finally:
        cap.release()
        log.info(f"📷 camera released ({addr})")


async def main():
    log.info(f"HairLens v2 (MediaPipe FaceMesh) — ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(stream_to_client, WS_HOST, WS_PORT,
                                max_size=10 * 1024 * 1024):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
