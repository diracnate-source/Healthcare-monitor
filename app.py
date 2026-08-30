import streamlit as st
import cv2
import numpy as np
import threading
import time
import os
import random
import logging
import json
import urllib.request
import mediapipe as mp
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from av import VideoFrame


# ============================================================
# 선별검사(SCREENING) 설정
# ============================================================

SCREENING_SECONDS = 7
SCREENING_MAX_FRAMES = 90  # 메모리 보호용 상한 (약 7초 * 15fps 여유분)


# ============================================================
# WebRTC 연결 설정
#
# Streamlit Community Cloud는 WebRTC 패킷(UDP)이 막히는 대표적인
# 환경이라, STUN만으로는 연결이 되지 않는 경우가 많다
# (streamlit-webrtc 공식 문서에 명시된 내용).
# 무료 공용 TURN(Open Relay)은 공식 문서에서도 "불안정하고 자주
# 다운된다"고 경고하므로, Twilio의 TURN 발급 API를 우선 사용하고,
# 설정이 안 되어 있으면 STUN만으로 폴백한다.
# ============================================================

@st.cache_resource(ttl=3600 * 20, show_spinner=False)
def _get_ice_servers_from_metered():

    app_name = st.secrets["METERED_APP_NAME"]
    api_key = st.secrets["METERED_API_KEY"]

    url = (
        f"https://{app_name}.metered.live/api/v1/turn/credentials"
        f"?apiKey={api_key}"
    )

    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data


def _get_ice_servers_from_twilio():

    from twilio.rest import Client

    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]

    client = Client(account_sid, auth_token)
    token = client.tokens.create()

    return token.ice_servers


def get_ice_servers():

    # 1순위: Metered.ca 개인 계정 (안정적, 무료 500MB/월)
    try:

        servers = _get_ice_servers_from_metered()

        print(f"[TURN] Metered.ca ICE 서버 발급 성공: {len(servers)}개")

        return servers, None

    except Exception as e_metered:

        metered_error = f"Metered: {type(e_metered).__name__}: {e_metered}"
        print(f"[TURN] Metered.ca 발급 실패: {metered_error}")

        # 2순위: Twilio (계정이 유료 전환되어 있다면 시도)
        try:

            servers = _get_ice_servers_from_twilio()

            print(f"[TURN] Twilio ICE 서버 발급 성공: {len(servers)}개")

            return servers, None

        except Exception as e_twilio:

            twilio_error = f"Twilio: {type(e_twilio).__name__}: {e_twilio}"
            print(f"[TURN] Twilio 발급도 실패: {twilio_error}")

            # 최종 폴백: STUN만 사용 (일부 환경에서 연결 불안정 가능)
            combined_error = f"{metered_error} / {twilio_error}"

            return (
                [{"urls": ["stun:stun.l.google.com:19302"]}],
                combined_error
            )


_ice_servers, _ice_error = get_ice_servers()

RTC_CONFIGURATION = {
    "iceServers": _ice_servers
}

if _ice_error and st.session_state.get("show_turn_debug", True):
    st.warning(
        f"⚠️ TURN 서버(Twilio) 연결 실패로 STUN만 사용 중입니다. "
        f"연결이 불안정할 수 있어요.\n\n디버그 정보: {_ice_error}"
    )

# aioice/aiortc의 시끄러운 INFO 로그(연결 종료 후 재시도 등)를 줄인다.
logging.getLogger("aioice").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 비접촉 선별검사 시스템",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 20px;
    }

    .pass {
        background: #e9fff1;
        border: 3px solid #00c853;
        color: #00863d;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
        margin: 15px 0;
    }

    .fail {
        background: #fff0f0;
        border: 3px solid #ff3030;
        color: #d00000;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
        margin: 15px 0;
    }

    .checking {
        background: #fff8df;
        border: 3px solid #ffb300;
        color: #946500;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        font-size: 22px;
        font-weight: 800;
        margin: 15px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 제목
# ============================================================

st.markdown(
    '<div class="title">🧠 AI 비접촉 선별검사 시스템</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">촬영 전 얼굴 및 촬영 조건 자동 검사</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "stage" not in st.session_state:
    st.session_state.stage = "precheck"

if "processor" not in st.session_state:
    st.session_state.processor = None

if "transitioning" not in st.session_state:
    st.session_state.transitioning = False

if "screening_processor" not in st.session_state:
    st.session_state.screening_processor = None

if "screening_result" not in st.session_state:
    st.session_state.screening_result = None

if "reaction_result" not in st.session_state:
    st.session_state.reaction_result = None

if "reaction_target_delay" not in st.session_state:
    st.session_state.reaction_target_delay = None

if "reaction_entry_time" not in st.session_state:
    st.session_state.reaction_entry_time = None

if "reaction_stimulus_time" not in st.session_state:
    st.session_state.reaction_stimulus_time = None

if "population_recorded" not in st.session_state:
    st.session_state.population_recorded = False


# ============================================================
# 초기화
# ============================================================

def reset_app():

    st.session_state.stage = "precheck"
    st.session_state.processor = None
    st.session_state.transitioning = False
    st.session_state.screening_processor = None
    st.session_state.screening_result = None
    st.session_state.reaction_result = None
    st.session_state.reaction_target_delay = None
    st.session_state.reaction_entry_time = None
    st.session_state.reaction_stimulus_time = None
    st.session_state.population_recorded = False


# ============================================================
# OpenCV 모델
#
# 방어 로직:
# 배포 환경(Streamlit Cloud 등)에서 numpy/opencv 버전 충돌이나
# 시스템 라이브러리 누락으로 cv2.data 자체가 없는 경우가 있다.
# 이 경우 원인을 화면에 명확히 표시하고 앱을 중단시킨다.
# ============================================================

if not hasattr(cv2, "data"):

    st.error(
        "OpenCV 모듈이 정상적으로 로드되지 않았습니다 "
        "(cv2.data 속성 없음).\n\n"
        f"설치된 OpenCV 버전: {getattr(cv2, '__version__', '알 수 없음')}\n\n"
        "requirements.txt의 numpy / opencv-python-headless 버전이 "
        "서로 호환되는지 확인하고, packages.txt에 시스템 라이브러리"
        "(libgl1, libglib2.0-0)가 포함되어 있는지 확인한 뒤 "
        "앱을 재배포(reboot)해 주세요."
    )
    st.stop()


FACE_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_XML = cv2.data.haarcascades + "haarcascade_eye.xml"
GLASSES_XML = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"


try:
    face_cascade = cv2.CascadeClassifier(FACE_XML)
    eye_cascade = cv2.CascadeClassifier(EYE_XML)
    glasses_cascade = cv2.CascadeClassifier(GLASSES_XML)
except Exception as e:
    st.error(f"Haar Cascade 모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()


if face_cascade.empty():
    st.error(
        "얼굴 검출 모델을 불러오지 못했습니다. "
        f"(경로: {FACE_XML})\n\n"
        "opencv-python-headless 설치가 손상되었을 수 있습니다. "
        "'Manage app' → 'Reboot app'으로 재배포해 보세요."
    )
    st.stop()


# ============================================================
# 안경 검출 (단일 프레임 raw 신호)
#
# 변경점:
# - 눈 스트립 전체가 아니라, eye_cascade로 검출된
#   "실제 눈 위치" 주변 패치에서만 안경테를 탐색한다.
#   (눈썹/코 그림자 등 눈과 무관한 영역에서의 오탐 감소)
# - 임계값을 기존보다 보수적으로 상향한다.
# ============================================================

def _score_glasses_patch(patch):
    """
    한 눈 주변 패치에서 안경 신호 3가지를 계산하고, 그 중 몇 개가
    임계값을 넘는지(0~3)를 반환한다. 세 조건을 모두(AND) 요구하는
    대신 다수결(2/3)로 판정해서 민감도를 높인다.
    """

    if patch.size == 0:
        return 0

    try:
        glasses_candidates = glasses_cascade.detectMultiScale(
            patch,
            scaleFactor=1.05,
            minNeighbors=6,   # 기존 10 → 완화 (너무 보수적이라 실측 미검출 다발)
            minSize=(25, 18)
        )
        glasses_count = len(glasses_candidates)
    except Exception:
        glasses_count = 0

    resized_patch = cv2.resize(patch, (160, 120))
    blurred_patch = cv2.GaussianBlur(resized_patch, (5, 5), 0)
    edges = cv2.Canny(blurred_patch, 60, 150)  # 임계값 소폭 완화

    edge_ratio = float(np.mean(edges > 0))

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=25,
        minLineLength=30,
        maxLineGap=5
    )

    horizontal_lines = 0

    if lines is not None:
        for line in lines[:, 0]:
            x1l, y1l, x2l, y2l = line
            angle = abs(np.degrees(np.arctan2(y2l - y1l, x2l - x1l)))
            if angle < 12 or angle > 168:
                horizontal_lines += 1

    signals = [
        glasses_count >= 1,
        horizontal_lines >= 2,   # 기존 3 → 완화
        edge_ratio >= 0.045,     # 기존 0.06 → 완화
    ]

    return sum(signals)


def detect_glasses_raw(eye_region, eyes):
    """
    NOTE(수정 이력): 기존 구현은 "haarcascade_eye.xml로 눈이 2개
    이상 검출되어야만" 안경 검사를 시도했다. 그런데 안경 렌즈의
    반사·테두리 때문에 바로 이 일반 눈 검출기가 눈을 못 찾는
    경우가 흔해서, "안경을 썼기 때문에 오히려 안경 미검출로
    처리되는" 역설이 있었다. 이를 고쳐 눈이 2개 미만 검출된
    경우에도 eye_region 전체를 안경 전용 cascade로 한 번 더
    스캔하도록 폴백을 추가했다. 또한 패치별 3개 신호를 모두
    요구(AND)하던 것을 다수결(2/3 이상)로 완화했다.
    """

    if eye_region.size == 0 or glasses_cascade.empty():
        return False

    eh_region, ew_region = eye_region.shape[:2]

    patches = []

    if eyes is not None and len(eyes) >= 2:

        eyes_sorted = sorted(eyes, key=lambda b: b[0])
        left_eye = eyes_sorted[0]
        right_eye = eyes_sorted[-1]

        for (ex, ey, ew, eh) in (left_eye, right_eye):

            pad_x = int(ew * 0.6)
            pad_y = int(eh * 0.8)

            x1 = max(0, ex - pad_x)
            x2 = min(ew_region, ex + ew + pad_x)
            y1 = max(0, ey - pad_y)
            y2 = min(eh_region, ey + eh + pad_y)

            patch = eye_region[y1:y2, x1:x2]

            if patch.size > 0:
                patches.append(patch)

    if not patches:
        # 폴백: 일반 눈 검출기가 눈을 찾지 못한 경우(안경 반사·
        # 테두리로 인한 미검출 가능성 포함) — eye_region 전체를
        # 좌/우 절반으로 나눠 안경 전용 cascade로 직접 스캔한다.
        mid = ew_region // 2
        left_half = eye_region[:, :mid]
        right_half = eye_region[:, mid:]

        for half in (left_half, right_half):
            if half.size > 0:
                patches.append(half)

    votes = 0

    for patch in patches:
        score = _score_glasses_patch(patch)
        # 3개 신호 중 2개 이상이면 이 패치는 "안경 의심"으로 카운트
        if score >= 2:
            votes += 1

    # 두 패치(또는 절반) 중 최소 한쪽에서 안경 의심 신호가 있으면
    # "이번 프레임에서 안경으로 의심됨" 으로 판정
    return votes >= 1


# ============================================================
# 얼굴 분석 (프레임 단위)
#
# 변경점:
# - 더 이상 이 함수 내부에서 "안경 → 즉시 FAIL"을 결정하지 않는다.
# - glasses 키에는 이번 프레임의 raw 신호만 담고,
#   최종 FAIL 여부는 VideoProcessor에서 최근 프레임들의
#   다수결(rolling majority)로 판단한다.
#   → 조명/각도로 인한 단발성 오탐이 곧바로 화면을 FAIL로
#     고정시키지 않도록 하여, 안경을 벗은 뒤에도 정상적으로
#     PASS 로 전환되게 한다.
# ============================================================

def analyze_frame(frame):

    result = {
        "status": "UNKNOWN",
        "message": "검사 중",

        "face": False,
        "face_box": None,
        "eyes": False,

        "glasses": False,   # 이번 프레임의 raw 신호 (다수결 이전)
        "mask": False,
        "hat": False,

        "center": False,
        "distance": False,
        "lighting": False,
        "frontal": False
    }

    if frame is None:
        return result


    h, w = frame.shape[:2]

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # 밝기
    # ========================================================

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    result["lighting"] = (
        45 <= brightness <= 220
        and
        contrast >= 18
    )


    # ========================================================
    # 얼굴 검출
    # ========================================================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(120, 120)
    )


    if len(faces) == 0:

        result["message"] = "얼굴을 정면으로 보여주세요."

        return result


    # 가장 큰 얼굴 사용
    x, y, fw, fh = max(
        faces,
        key=lambda box: box[2] * box[3]
    )


    result["face"] = True
    result["face_box"] = (int(x), int(y), int(fw), int(fh))


    # ========================================================
    # 얼굴 거리
    # ========================================================

    face_ratio = fw / float(w)

    result["distance"] = (
        0.25 <= face_ratio <= 0.72
    )


    # ========================================================
    # 중앙 위치
    # ========================================================

    face_center_x = x + fw / 2
    frame_center_x = w / 2

    center_error = (
        abs(face_center_x - frame_center_x) / w
    )

    result["center"] = (
        center_error <= 0.15
    )


    # ========================================================
    # 얼굴 ROI
    # ========================================================

    roi = gray[
        y:y + fh,
        x:x + fw
    ]

    if roi.size == 0:
        return result


    # ========================================================
    # 조명 (얼굴 영역 기준으로 재판정)
    #
    # 화면 전체 밝기로 판정하면 배경이 밝은 역광 상황에서
    # 얼굴이 어두워도 "조명 정상"으로 잘못 통과하거나,
    # 반대로 밝은 배경 때문에 전체 평균이 튀어 계속 FAIL 나는
    # 문제가 있어 얼굴 ROI 기준으로 다시 계산한다.
    # ========================================================

    face_brightness = float(np.mean(roi))
    face_contrast = float(np.std(roi))

    result["lighting"] = (
        40 <= face_brightness <= 225
        and
        face_contrast >= 15
    )


    # ========================================================
    # 눈 영역
    # ========================================================

    eye_top = int(fh * 0.18)
    eye_bottom = int(fh * 0.58)

    eye_region = roi[
        eye_top:eye_bottom,
        :
    ]


    eyes = []

    if eye_region.size > 0:

        eyes = eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(15, 15)
        )


    # 양쪽 눈이 충분히 검출되어야 PASS
    result["eyes"] = len(eyes) >= 2


    # ========================================================
    # 정면 여부
    # ========================================================

    frontal = False

    if len(eyes) >= 2:

        eye_centers = []

        for ex, ey, ew, eh in eyes:

            eye_centers.append(
                (
                    ex + ew / 2,
                    ey + eh / 2
                )
            )


        eye_centers.sort(
            key=lambda p: p[0]
        )


        left_eye = eye_centers[0]
        right_eye = eye_centers[-1]


        eye_distance = abs(
            right_eye[0] - left_eye[0]
        )

        eye_height_difference = abs(
            right_eye[1] - left_eye[1]
        )


        if (
            eye_distance > fw * 0.18
            and
            eye_height_difference <
            eye_distance * 0.35
        ):

            frontal = True


    result["frontal"] = frontal


    # ========================================================
    # 안경 검사 (raw 신호만 계산, 최종 판정은 processor에서)
    # ========================================================

    result["glasses"] = detect_glasses_raw(eye_region, eyes)


    # ========================================================
    # 마스크 검사
    #
    # 경량 휴리스틱.
    # 불확실하면 FAIL이 아니라 UNKNOWN으로 처리.
    # ========================================================

    lower_top = int(fh * 0.52)
    lower_bottom = int(fh * 0.92)

    lower_face = roi[
        lower_top:lower_bottom,
        :
    ]


    mask_suspicious = False


    if lower_face.size > 0:

        lower_std = float(
            np.std(lower_face)
        )

        lower_mean = float(
            np.mean(lower_face)
        )


        # 얼굴 하단이 지나치게 균일한 경우
        # 마스크 후보로만 사용
        mask_suspicious = (
            lower_std < 22
            and
            55 < lower_mean < 225
        )


    # 눈은 보이는데 하단만 지나치게 균일한 경우
    if (
        mask_suspicious
        and
        result["eyes"]
    ):

        result["mask"] = False


    # ========================================================
    # 모자 검사
    #
    # 매우 보수적으로 후보만 계산
    # ========================================================

    upper_bottom = int(
        fh * 0.22
    )

    upper_face = roi[
        0:upper_bottom,
        :
    ]


    hat_suspicious = False


    if upper_face.size > 0:

        upper_edges = cv2.Canny(
            upper_face,
            60,
            150
        )

        upper_edge_ratio = float(
            np.mean(
                upper_edges > 0
            )
        )

        hat_suspicious = (
            upper_edge_ratio > 0.25
        )


    # 모자는 현재 확실한 AI 검출기가 아니므로
    # 단순 edge만으로 FAIL 처리하지 않는다.
    result["hat"] = False


    # ========================================================
    # 모든 촬영 조건
    #
    # 참고: 안경 여부는 여기서 판정하지 않는다.
    # (VideoProcessor의 rolling majority 결과가 최종 FAIL/PASS를 덮어씀)
    # ========================================================

    required = [
        result["face"],
        result["eyes"],
        result["center"],
        result["distance"],
        result["lighting"],
        result["frontal"]
    ]


    if all(required):

        result["status"] = "PASS"

        result["message"] = (
            "촬영 조건이 완료되었습니다."
        )

        return result


    # ========================================================
    # 미충족 이유
    # ========================================================

    result["status"] = "UNKNOWN"


    if not result["face"]:

        result["message"] = (
            "얼굴을 카메라 정면에 보여주세요."
        )

    elif not result["eyes"]:

        result["message"] = (
            "양쪽 눈이 모두 보이도록 해주세요."
        )

    elif not result["center"]:

        result["message"] = (
            "얼굴을 화면 중앙에 맞춰주세요."
        )

    elif not result["distance"]:

        result["message"] = (
            "카메라와의 거리를 조절해주세요."
        )

    elif not result["lighting"]:

        result["message"] = (
            "조명이 너무 어둡거나 밝습니다."
        )

    elif not result["frontal"]:

        result["message"] = (
            "정면을 바라봐 주세요."
        )

    return result


# ============================================================
# VIDEO PROCESSOR
# ============================================================

# ============================================================
# 선별검사 - mediapipe 랜드마커 (앱 수명 동안 1회만 로드/다운로드)
# ============================================================

@st.cache_resource(show_spinner="랜드마크 분석 모델을 불러오는 중...")
def get_face_landmarker():

    model_path = "face_landmarker.task"

    if not os.path.exists(model_path):

        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"랜드마크 모델 다운로드에 실패했습니다: {e}")
            st.stop()

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    return FaceLandmarker.create_from_options(options)


# ============================================================
# 선별검사 - 프레임 버퍼 캡처용 프로세서
# ============================================================

class ScreeningProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()
        self.start_time = time.time()
        self.frames = []
        self.done = False

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        h, w = image.shape[:2]
        max_width = 480

        if w > max_width:
            scale = max_width / float(w)
            image = cv2.resize(
                image,
                (max_width, int(h * scale))
            )

        with self.lock:

            elapsed = time.time() - self.start_time

            if (
                elapsed <= SCREENING_SECONDS
                and len(self.frames) < SCREENING_MAX_FRAMES
            ):
                self.frames.append(image.copy())

            if elapsed > SCREENING_SECONDS:
                self.done = True

        remaining = max(
            0.0,
            SCREENING_SECONDS - (time.time() - self.start_time)
        )

        cv2.rectangle(image, (10, 10), (300, 60), (0, 0, 0), -1)

        cv2.putText(
            image,
            f"REC {remaining:0.1f}s",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        return VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    def get_state(self):

        with self.lock:

            elapsed = time.time() - self.start_time
            frame_count = len(self.frames)
            done = self.done
            frames_copy = list(self.frames) if done else None

        return elapsed, frame_count, done, frames_copy


# ============================================================
# 선별검사 - 랜드마크 시각화 및 지표 산출
# ============================================================

# ============================================================
# 선별검사 - 랜드마크 인덱스 상수
#
# 참고: MediaPipe FaceLandmarker는 478개 랜드마크(얼굴 468개 +
# 양쪽 눈동자(iris) 각 5개)를 반환한다. 아래 인덱스는 공식
# MediaPipe Face Mesh 문서 기준의 근사 매핑이며, 임상 검증된
# 좌표가 아니라 프로토타입 계산용이다.
# ============================================================

LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

LEFT_IRIS_IDX = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]

NOSE_TIP_IDX = 1
LEFT_MOUTH_IDX = 61
RIGHT_MOUTH_IDX = 291
LEFT_BROW_IDX = 105
RIGHT_BROW_IDX = 334

EAR_BLINK_THRESHOLD = 0.21  # 이 값 아래로 내려가면 "눈 감음"으로 판정


def _eye_aspect_ratio(pts, idx):

    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]

    vertical = (
        np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    )
    horizontal = np.linalg.norm(p1 - p4) * 2.0

    if horizontal == 0:
        return 0.3

    return float(vertical / horizontal)


def generate_landmark_visualization_and_metrics(frames):

    landmarker = get_face_landmarker()

    target_frame = frames[len(frames) // 2].copy()
    h, w = target_frame.shape[:2]

    # 최대 60프레임까지만 분석 (성능 보호)
    ANALYSIS_MAX_SAMPLES = 60
    step = max(1, len(frames) // ANALYSIS_MAX_SAMPLES)

    motion_deltas = []
    ear_series = []
    asymmetry_series = []
    gaze_x_series = []

    prev_pts = None
    has_iris = None  # 첫 검출 시 478점 여부 확인

    for idx in range(0, len(frames), step):

        f_rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)
        res = landmarker.detect(mp_image)

        if not res.face_landmarks:
            continue

        pts = np.array(
            [(lm.x, lm.y) for lm in res.face_landmarks[0]]
        )

        if has_iris is None:
            has_iris = len(pts) >= 478

        # --- 표정 변화 / 미세 근육 움직임 (프레임 간 랜드마크 이동량) ---
        if prev_pts is not None and prev_pts.shape == pts.shape:
            motion_deltas.append(
                float(np.mean(np.abs(pts - prev_pts)))
            )
        prev_pts = pts

        # --- 눈 깜빡임 (EAR) ---
        left_ear = _eye_aspect_ratio(pts, LEFT_EYE_EAR_IDX)
        right_ear = _eye_aspect_ratio(pts, RIGHT_EYE_EAR_IDX)
        ear_series.append((left_ear + right_ear) / 2.0)

        # --- 안면 비대칭 ---
        nose = pts[NOSE_TIP_IDX]

        mouth_l_dist = np.linalg.norm(pts[LEFT_MOUTH_IDX] - nose)
        mouth_r_dist = np.linalg.norm(pts[RIGHT_MOUTH_IDX] - nose)
        brow_l_dist = np.linalg.norm(pts[LEFT_BROW_IDX] - nose)
        brow_r_dist = np.linalg.norm(pts[RIGHT_BROW_IDX] - nose)

        mouth_asym = (
            abs(mouth_l_dist - mouth_r_dist)
            / (mouth_l_dist + mouth_r_dist + 1e-6)
        )
        brow_asym = (
            abs(brow_l_dist - brow_r_dist)
            / (brow_l_dist + brow_r_dist + 1e-6)
        )

        asymmetry_series.append((mouth_asym + brow_asym) / 2.0)

        # --- 시선 이동 (iris가 있을 때만) ---
        if has_iris:

            left_iris = pts[LEFT_IRIS_IDX].mean(axis=0)
            right_iris = pts[RIGHT_IRIS_IDX].mean(axis=0)

            gaze_x_series.append(
                float((left_iris[0] + right_iris[0]) / 2.0)
            )

    # ------------------------------------------------------------
    # 중간 프레임에 랜드마크 시각화
    # ------------------------------------------------------------

    target_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    mp_target = mp.Image(image_format=mp.ImageFormat.SRGB, data=target_rgb)
    detection_result = landmarker.detect(mp_target)

    face_detected = bool(detection_result.face_landmarks)

    if face_detected:

        for face_landmarks in detection_result.face_landmarks:
            for lm in face_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(target_frame, (x, y), 1, (0, 255, 0), -1)

    annotated_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------
    # 지표 집계
    # ------------------------------------------------------------

    expression_change = (
        float(np.mean(motion_deltas) * 1000) if motion_deltas else 0.0
    )
    micro_movement = (
        float(np.std(motion_deltas) * 500) if motion_deltas else 0.0
    )

    facial_asymmetry = (
        float(np.mean(asymmetry_series) * 1000) if asymmetry_series else 0.0
    )

    if len(ear_series) >= 2:
        below = np.array(ear_series) < EAR_BLINK_THRESHOLD
        blink_count = int(np.sum(below[1:] & ~below[:-1]))
        duration_sec = max(1.0, SCREENING_SECONDS)
        blink_rate = float(blink_count / duration_sec * 60.0)
    else:
        blink_rate = 0.0

    if has_iris and len(gaze_x_series) >= 2:
        gaze_variability = float(np.std(gaze_x_series) * 1000)
    else:
        gaze_variability = None  # iris 미검출 시 측정 불가

    metrics = {
        "expression_change": expression_change,
        "micro_movement": micro_movement,
        "facial_asymmetry": facial_asymmetry,
        "blink_rate": blink_rate,
        "gaze_variability": gaze_variability,
        "iris_available": bool(has_iris),
    }

    return annotated_rgb, metrics, face_detected


# ============================================================
# 개인 기준선(personal baseline) + 모집단 기준(population baseline)
# 저장/조회
#
# PoC 단계 임시 저장소: 서버 로컬 JSON 파일에 저장한다.
# 정식 서비스 단계에서는 반드시 실제 DB로 교체해야 한다
# (Streamlit Cloud 재배포 시 파일이 초기화될 수 있음).
#
# 개인 저장소(baseline_store.json): user_id별 누적 세션 수와
# 지표별 러닝 평균(running mean)을 저장 — compute_risk_score의
# 개인화(personalization) 항에 쓰인다. 화면에 보여주는 "직전
# 측정치 대비 비교"용으로 가장 최근 1회 측정치도 함께 보관한다.
#
# 모집단 저장소(population_store.json): 지금까지 앱을 사용한
# 모든 세션의 지표값을 Welford's online algorithm으로 누적해
# 지표별 평균·표준편차를 계속 갱신한다 — 세션이 쌓일수록 "평소
# 범위" 추정이 스스로 정교해지는 구조.
# ============================================================

BASELINE_STORE_PATH = "baseline_store.json"
POPULATION_STORE_PATH = "population_store.json"
_baseline_lock = threading.Lock()
_population_lock = threading.Lock()

# 모집단 표본이 이 값 미만인 지표는 온라인 추정치(평균·표준편차)가
# 통계적으로 불안정하다. 이럴 때 "그럴듯한 평균값"을 지어내는 대신
# — 그 자체가 우리가 피하려던 '근거 없는 상수' 문제이므로 —
# 지표별 표본이 충분해지기 전까지는 기존 검증된 레거시 선형 가중합
# 방식(원래 app.py의 최초 버전)으로 안전하게 폴백한다.
MIN_POPULATION_N = 30

# 레거시(부트스트랩) 모드에서 쓰는 고정 계수 — 최초 프로토타입의
# 값을 그대로 유지한다. 이 값들 역시 임상 검증된 것은 아니며,
# 모집단 데이터가 쌓이기 전까지 쓰는 임시 방편임을 명시한다.
LEGACY_WEIGHTS = {
    "expression_change": 0.6,
    "micro_movement": 0.8,
    "facial_asymmetry": 0.5,
    "blink_rate": 1.0,
    "gaze_variability": 0.4,
}

# 지표별 가중치(데이터 기반 모드에서 사용) — 문헌에서 반복 검증된
# 정도를 반영한 잠정값.
# (반응속도·눈깜빡임·안면비대칭: 비교적 반복검증된 편 → 1.0,
#  표정변화·시선변동성: 중간 → 0.8, 미세근육움직임: 근거 약함 → 0.5)
# NOTE: 이 가중치 역시 사람이 정한 값으로, 실제 라벨(임상 진단·
# MMSE 등) 데이터로 로지스틱 회귀 등을 돌려야 객관적 가중치를
# 얻을 수 있다. 현재는 그런 라벨 데이터가 없다.
METRIC_WEIGHTS = {
    "expression_change": 0.8,
    "micro_movement": 0.5,
    "facial_asymmetry": 1.0,
    "blink_rate": 1.0,
    "gaze_variability": 0.8,
    "reaction_ms": 1.0,
}

METRIC_LABELS = {
    "expression_change": "표정 변화",
    "micro_movement": "미세 근육 움직임",
    "facial_asymmetry": "안면 비대칭",
    "blink_rate": "눈 깜빡임 빈도(회/분)",
    "gaze_variability": "시선 이동 변동성",
    "reaction_ms": "반응속도(ms)",
}

# empirical Bayes 개인화 가중치 w = n_personal / (n_personal + K)
EMPIRICAL_BAYES_K = 4

# 비선형 스쿼싱 risk_score = 100 * (1 - exp(-Z/LAMBDA)) 의 상수
SQUASH_LAMBDA = 2.0

# 3단계 완충 라벨 임계값 (통합 Z 기준, 데이터 기반 모드에서 사용)
TIER_THRESHOLDS = {
    "양호": 0.5,
    "주의": 1.5,
    # 그 이상은 "확인 권장"
}

# 3단계 완충 라벨 임계값 (레거시 부트스트랩 모드, 0~100 스코어 기준)
LEGACY_TIER_THRESHOLDS = {
    "양호": 33.0,
    "주의": 66.0,
    # 그 이상은 "확인 권장"
}


def _load_json_store(path, lock):

    with lock:

        if not os.path.exists(path):
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def _save_json_store(path, lock, store):

    with lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 모집단 통계 (population baseline)
# ------------------------------------------------------------

def get_population_stats():
    """
    지표별 (n, mean, std, ready)를 반환한다. ready=True는 표본이
    MIN_POPULATION_N 이상이라 온라인 추정치를 신뢰할 수 있다는 뜻.
    표본이 없거나 부족하면 mean/std는 None이며(지어낸 값을 채우지
    않음), 호출 측(compute_risk_score)이 이를 보고 레거시 방식으로
    폴백한다.
    """

    store = _load_json_store(POPULATION_STORE_PATH, _population_lock)

    stats = {}

    for key in LEGACY_WEIGHTS.keys() | {"reaction_ms"}:

        entry = store.get(key)
        n = entry.get("n", 0) if entry else 0

        if entry is None or n < 2:
            stats[key] = {"n": n, "mean": None, "std": None, "ready": False}
            continue

        mean = entry["mean"]
        # Welford: variance = M2 / (n - 1)
        variance = entry["m2"] / max(1, n - 1)
        std = float(np.sqrt(variance)) if variance > 0 else None

        stats[key] = {
            "n": n,
            "mean": mean,
            "std": std,
            "ready": (n >= MIN_POPULATION_N and std is not None and std > 1e-9),
        }

    return stats


def update_population_stats(metrics, reaction_ms=None):
    """
    Welford's online algorithm으로 지표별 평균·분산을 갱신한다.
    측정되지 않은(None) 지표는 건너뛴다.
    """

    values = {
        "expression_change": metrics.get("expression_change"),
        "micro_movement": metrics.get("micro_movement"),
        "facial_asymmetry": metrics.get("facial_asymmetry"),
        "blink_rate": metrics.get("blink_rate"),
        "gaze_variability": metrics.get("gaze_variability"),
        "reaction_ms": reaction_ms,
    }

    with _population_lock:

        store = _load_json_store(POPULATION_STORE_PATH, threading.Lock())

        for key, x in values.items():

            if x is None:
                continue

            entry = store.get(key, {"n": 0, "mean": 0.0, "m2": 0.0})

            n = entry["n"] + 1
            delta = x - entry["mean"]
            mean = entry["mean"] + delta / n
            delta2 = x - mean
            m2 = entry["m2"] + delta * delta2

            store[key] = {"n": n, "mean": mean, "m2": m2}

        with open(POPULATION_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 개인 기준선 (personal baseline)
# ------------------------------------------------------------

def get_baseline(user_id):
    """직전 측정치 비교 화면(표시용)을 위한 개인 기준선 조회."""

    store = _load_json_store(BASELINE_STORE_PATH, _baseline_lock)
    entry = store.get(user_id)

    if entry is None:
        return None

    # 과거 스키마(metrics/risk_score만 저장) 호환
    return {
        "metrics": entry.get("metrics", entry.get("last_metrics")),
        "risk_score": entry.get("risk_score", entry.get("last_risk_score", 0.0)),
        "saved_at": entry.get("saved_at", ""),
    }


def get_personal_stats(user_id):
    """
    개인화 항에 쓰이는 (n_personal, personal_mean) 조회.
    기록이 없으면 n_personal=0.
    """

    if not user_id:
        return 0, {}

    store = _load_json_store(BASELINE_STORE_PATH, _baseline_lock)
    entry = store.get(user_id)

    if entry is None:
        return 0, {}

    n_personal = entry.get("n_sessions", 1 if "metrics" in entry else 0)
    personal_means = entry.get("metric_means", {})

    return n_personal, personal_means


def save_baseline(user_id, metrics, risk_score, reaction_ms=None):
    """
    표시용 최근 측정치를 갱신하고, 개인화 항에 쓰이는 지표별
    러닝 평균(metric_means)과 세션 수(n_sessions)를 누적한다.
    """

    values = {
        "expression_change": metrics.get("expression_change"),
        "micro_movement": metrics.get("micro_movement"),
        "facial_asymmetry": metrics.get("facial_asymmetry"),
        "blink_rate": metrics.get("blink_rate"),
        "gaze_variability": metrics.get("gaze_variability"),
        "reaction_ms": reaction_ms,
    }

    with _baseline_lock:

        store = _load_json_store(BASELINE_STORE_PATH, threading.Lock())
        entry = store.get(user_id, {"n_sessions": 0, "metric_means": {}})

        n_prev = entry.get("n_sessions", 0)
        means = dict(entry.get("metric_means", {}))

        for key, x in values.items():
            if x is None:
                continue
            prev_mean = means.get(key, x)
            # 세션 단위 러닝 평균 갱신 (지표별로 결측이 섞여도
            # 독립적으로 카운트되도록 각 지표 자체 관측 횟수는
            # 별도 관리하지 않고, 전체 세션 수 n_prev+1로 근사한다.
            means[key] = prev_mean + (x - prev_mean) / (n_prev + 1)

        store[user_id] = {
            "metrics": metrics,              # 최근 1회 측정치(표시용)
            "risk_score": risk_score,        # 최근 1회 스코어(표시용)
            "saved_at": time.strftime("%Y-%m-%d %H:%M"),
            "n_sessions": n_prev + 1,
            "metric_means": means,
        }

        with open(BASELINE_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 위험도 통합 공식 v2 — 개인화 베이지안 Z-score 스코어
# ------------------------------------------------------------

def compute_risk_score(metrics, reaction_ms=None, user_id=None):
    """
    NOTE: 이 알고리즘은 기존 방식(임의 계수 곱 → 단순 평균 → 선형
    clip)의 한계를 통계적으로 좀 더 설명 가능한 구조로 바꾼 것이며,
    실제 임상 라벨(진단·MMSE 등) 데이터로 성능이 검증된 것은
    아니다. PoC로 라벨 데이터가 모이기 전까지는 여전히 '검증되지
    않은 프로토타입' 스코어라는 점을 화면에 항상 함께 고지해야
    한다.

    두 가지 모드로 동작한다.

    [부트스트랩 모드] 모집단 표본이 MIN_POPULATION_N개 미만인 경우.
      "평균이 얼마다"를 지어낼 근거가 없으므로, 억지로 Z-score를
      계산하는 대신 기존에 쓰던 검증되지 않은 고정 계수(LEGACY_WEIGHTS)
      선형 가중합 방식으로 안전하게 계산한다. 데이터가 부족한
      상태에서 '통계적으로 보이는' 숫자를 지어내는 것보다, 이렇게
      명시적으로 표시되는 임시 방식을 쓰는 편이 더 정직하다.

    [데이터 기반 모드] 모집단 표본이 충분히 쌓인 경우.
      1) 모집단 기준(μ_pop, σ_pop) — 실측 온라인 통계
      2) 개인 기준(μ_personal) — 없으면 모집단에 전적으로 의존
      3) 경험적 베이즈로 두 기준을 세션 수 기반 가중 결합
      4) Z-score 표준화 후, "평소보다 나빠진" 방향만 인정(one-sided)
      5) 지표별 가중 결합으로 통합 Z 산출
      6) 지수함수 스쿼싱으로 0~100 스코어 산출 (선형 clip 대체)
      7) 통합 Z를 3단계 라벨(양호/주의/확인 권장)로 변환
    """

    raw_values = {
        "expression_change": metrics.get("expression_change"),
        "micro_movement": metrics.get("micro_movement"),
        "facial_asymmetry": metrics.get("facial_asymmetry"),
        "blink_rate": metrics.get("blink_rate"),
        "gaze_variability": metrics.get("gaze_variability"),
        "reaction_ms": reaction_ms,
    }

    pop_stats = get_population_stats()
    population_n = {k: pop_stats[k]["n"] for k in pop_stats}

    # 게이트 지표: 촬영이 성공하면 항상 계산되는 blink_rate의 표본
    # 수로 전체 모드를 결정한다(부분적으로 z와 레거시를 섞으면
    # 스케일이 달라 해석이 어려워지므로, 모드는 세션 전체에 대해
    # 하나로 통일한다).
    bootstrap_mode = not pop_stats["blink_rate"]["ready"]

    if bootstrap_mode:

        raw_scores = {}

        for key, weight in LEGACY_WEIGHTS.items():
            x = raw_values[key]
            label = METRIC_LABELS[key]
            raw_scores[label] = round(x * weight, 2) if x is not None else None

        if reaction_ms is not None:
            raw_scores["반응속도(ms)"] = round(
                max(0.0, (reaction_ms - 300.0) * 0.05), 2
            )
        else:
            raw_scores["반응속도(ms)"] = None

        scoreable = [v for v in raw_scores.values() if v is not None]

        risk_score = float(
            np.clip(np.mean(scoreable) * 1.2, 0.0, 100.0)
        ) if scoreable else 0.0

        if risk_score < LEGACY_TIER_THRESHOLDS["양호"]:
            tier = "양호"
        elif risk_score < LEGACY_TIER_THRESHOLDS["주의"]:
            tier = "주의"
        else:
            tier = "확인 권장"

        detail = {
            "mode": "bootstrap_legacy",
            "combined_z": None,
            "tier": tier,
            "n_personal": 0,
            "population_n": population_n,
        }

        return risk_score, raw_scores, detail

    # ---------------- 데이터 기반 모드 ----------------

    n_personal, personal_means = get_personal_stats(user_id)

    z_scores = {}
    raw_scores = {}  # 화면 표시용 (지표별 z+ 값)

    weighted_sum = 0.0
    weight_total = 0.0

    for key, x in raw_values.items():

        label = METRIC_LABELS[key]

        if x is None:
            raw_scores[label] = None
            continue

        pop = pop_stats[key]

        if not pop["ready"]:
            # 이 지표만 아직 표본이 부족한 경우(예: 시선/반응속도는
            # 선택적으로만 측정됨) — 통합 스코어 계산에서는 제외하고
            # 화면에는 '데이터 축적 중'으로 표시한다.
            raw_scores[label] = None
            continue

        # 경험적 베이즈 가중 결합: 개인 세션이 쌓일수록 개인 기준
        # 쪽으로 자연스럽게 수렴한다.
        w_personal = n_personal / (n_personal + EMPIRICAL_BAYES_K)
        mu_personal = personal_means.get(key, pop["mean"])
        mu_ref = w_personal * mu_personal + (1 - w_personal) * pop["mean"]
        sigma_ref = pop["std"]

        z = (x - mu_ref) / sigma_ref
        z_plus = max(0.0, z)  # 평소보다 좋아진 방향은 위험 가산 없음

        z_scores[key] = z_plus

        weight = METRIC_WEIGHTS[key]
        weighted_sum += weight * z_plus
        weight_total += weight

        raw_scores[label] = round(z_plus, 3)  # 화면에는 z+ 값을 표시

    combined_z = (weighted_sum / weight_total) if weight_total > 0 else 0.0

    risk_score = float(100.0 * (1.0 - np.exp(-combined_z / SQUASH_LAMBDA)))
    risk_score = float(np.clip(risk_score, 0.0, 100.0))

    if combined_z < TIER_THRESHOLDS["양호"]:
        tier = "양호"
    elif combined_z < TIER_THRESHOLDS["주의"]:
        tier = "주의"
    else:
        tier = "확인 권장"

    detail = {
        "mode": "data_driven",
        "combined_z": combined_z,
        "tier": tier,
        "n_personal": n_personal,
        "population_n": population_n,
    }

    return risk_score, raw_scores, detail


class PrecheckProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.latest_result = None

        self.pass_history = deque(
            maxlen=10
        )

        self.fail_history = deque(
            maxlen=5
        )

        # 안경 raw 신호의 최근 이력 (다수결 판정용)
        # maxlen=8 -> 대략 초당 15~30프레임 기준 0.3~0.5초 내
        # 안경을 벗으면 히스토리가 대부분 갱신되어 빠르게 복구된다.
        self.glasses_history = deque(
            maxlen=8
        )

        self.frame_count = 0

        # 매 프레임 전체 분석을 돌리면 CPU 부하로 영상이 느려지므로
        # N프레임마다 한 번만 무거운 분석(cascade + Canny + Hough)을 수행한다.
        self._frame_idx = 0
        self._analyze_every_n = 2


    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # ----------------------------------------------
        # 해상도 축소
        #
        # 브라우저가 media_stream_constraints를 무시하고 고해상도로
        # 캡처하는 경우를 대비해, 분석/표시용 프레임 자체를 축소한다.
        # Haar cascade는 픽셀 수에 거의 선형으로 느려지므로 이 한 번의
        # resize가 전체 지연을 크게 줄여준다.
        # ----------------------------------------------

        h, w = image.shape[:2]
        max_width = 480

        if w > max_width:

            scale = max_width / float(w)

            image = cv2.resize(
                image,
                (max_width, int(h * scale))
            )


        self._frame_idx += 1

        analyze_this_frame = (
            self._frame_idx % self._analyze_every_n == 0
            or self.latest_result is None
        )


        if analyze_this_frame:

            result = analyze_frame(
                image
            )

            with self.lock:

                # ----------------------------------------------
                # 안경 다수결 판정
                #
                # 최근 8프레임 중 5프레임 이상에서 안경 신호가
                # 잡혀야만 "실제로 안경을 착용 중"으로 확정한다.
                # 단발성 오탐(그림자, 눈썹 등)은 무시된다.
                # ----------------------------------------------

                self.glasses_history.append(
                    result["glasses"]
                )

                glasses_votes = sum(self.glasses_history)
                glasses_total = len(self.glasses_history)

                glasses_confirmed = (
                    glasses_total >= 4
                    and glasses_votes / glasses_total >= 0.6
                )

                if glasses_confirmed:

                    result["status"] = "FAIL"

                    result["message"] = (
                        "안경 또는 선글라스가 감지되었습니다. "
                        "안경을 벗어주세요."
                    )

                self.latest_result = result

                self.frame_count += 1

                self.pass_history.append(
                    result["status"] == "PASS"
                )

                self.fail_history.append(
                    result["status"] == "FAIL"
                )

        else:

            # 이번 프레임은 분석을 건너뛰고 최근 결과를 재사용한다
            # (화면 오버레이만 매 프레임 갱신되어 영상 자체는 끊기지 않는다)
            with self.lock:
                result = self.latest_result


        # ====================================================
        # 화면 표시
        # ====================================================

        if result["status"] == "PASS":

            display_color = (
                0,
                220,
                0
            )

            display_text = (
                "PASS - READY"
            )

        elif result["status"] == "FAIL":

            display_color = (
                0,
                0,
                255
            )

            display_text = (
                "NOT ELIGIBLE"
            )

        else:

            display_color = (
                0,
                180,
                255
            )

            display_text = (
                "CHECKING"
            )


        # 얼굴 위치 표시
        #
        # analyze_frame()이 이미 계산해 둔 좌표(result["face_box"])를
        # 재사용한다. 예전 코드는 화면에 사각형을 그리기 위해 cascade를
        # 프레임당 한 번 더(사실상 2배) 실행하고 있었다 — 이것이 영상
        # 지연의 큰 원인 중 하나였다.

        face_box = result.get("face_box") if result else None

        if face_box is not None:

            x, y, fw, fh = face_box

            cv2.rectangle(
                image,
                (x, y),
                (x + fw, y + fh),
                display_color,
                3
            )


        # 상태 박스
        cv2.rectangle(
            image,
            (10, 10),
            (430, 65),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            image,
            display_text,
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            display_color,
            2,
            cv2.LINE_AA
        )


        return VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


    # ========================================================
    # 상태 가져오기
    # ========================================================

    def get_state(self):

        with self.lock:

            result = self.latest_result

            pass_count = sum(
                self.pass_history
            )

            fail_count = sum(
                self.fail_history
            )

            frame_count = self.frame_count


        return (
            result,
            pass_count,
            fail_count,
            frame_count
        )


# ============================================================
# PRECHECK
# ============================================================

if st.session_state.stage == "precheck":

    st.info(
        "촬영 전 안경·선글라스·마스크·모자를 제거하고 "
        "정면을 바라봐 주세요."
    )


    st.caption(
        "조건이 일정 시간 연속으로 충족되면 "
        "촬영 버튼 없이 자동으로 선별검사로 이동합니다."
    )


    # ========================================================
    # 카메라
    # ========================================================

    ctx = webrtc_streamer(

        key="health-precheck-camera-v6",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=PrecheckProcessor,

        media_stream_constraints={
            "video": {
                "width": {"ideal": 480, "max": 640},
                "height": {"ideal": 360, "max": 480},
                "frameRate": {"ideal": 15, "max": 20}
            },
            "audio": False
        },

        async_processing=True,

        rtc_configuration=RTC_CONFIGURATION
    )


    # processor 저장
    if ctx.video_processor is not None:

        st.session_state.processor = (
            ctx.video_processor
        )


    # ========================================================
    # 자동 상태 감시
    # ========================================================

    @st.fragment(run_every=0.5)
    def monitor_camera():

        processor = (
            st.session_state.get(
                "processor"
            )
        )


        if processor is None:

            st.markdown(
                '<div class="checking">'
                '📷 카메라 준비 중...'
                '</div>',
                unsafe_allow_html=True
            )

            return


        result, pass_count, fail_count, frame_count = (
            processor.get_state()
        )


        # ====================================================
        # PASS
        # ====================================================

        if pass_count >= 8:

            st.markdown(
                '<div class="pass">'
                '✅ 촬영 조건 완료'
                '</div>',
                unsafe_allow_html=True
            )


            st.success(
                "촬영 조건이 완료되었습니다. "
                "자동으로 선별검사를 시작합니다."
            )


            if not st.session_state.transitioning:

                st.session_state.transitioning = True
                st.session_state.stage = "screening"

                # 전체 앱 다시 실행
                st.rerun()


            return


        # ====================================================
        # FAIL
        # ====================================================

        if fail_count >= 2:

            st.markdown(
                '<div class="fail">'
                '❌ 부적합 — 촬영 불가'
                '</div>',
                unsafe_allow_html=True
            )


            if result:

                st.error(
                    result.get(
                        "message",
                        "촬영 조건을 충족하지 못했습니다."
                    )
                )


            return


        # ====================================================
        # 검사 중
        # ====================================================

        st.markdown(
            '<div class="checking">'
            '🔎 검사 중 — 촬영 불가'
            '</div>',
            unsafe_allow_html=True
        )


        if result:

            st.warning(
                result.get(
                    "message",
                    "촬영 조건을 확인하고 있습니다."
                )
            )


        st.caption(
            f"연속 PASS: {pass_count}/8"
        )

        st.caption(
            f"검사 프레임: {frame_count}"
        )


    monitor_camera()


# ============================================================
# SCREENING
# ============================================================

elif st.session_state.stage == "screening":

    st.markdown(
        """
        <div class="pass">
        🧠 선별검사를 시작합니다.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # 아직 분석 결과가 없으면 -> 촬영 진행
    # ========================================================

    if st.session_state.screening_result is None:

        st.info(
            f"화면을 정면으로 바라봐 주세요. "
            f"{SCREENING_SECONDS}초간 자동으로 촬영됩니다."
        )

        ctx = webrtc_streamer(

            key="health-screening-camera-v1",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=ScreeningProcessor,

            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480, "max": 640},
                    "height": {"ideal": 360, "max": 480},
                    "frameRate": {"ideal": 15, "max": 20}
                },
                "audio": False
            },

            async_processing=True,

            rtc_configuration=RTC_CONFIGURATION
        )


        if ctx.video_processor is not None:

            st.session_state.screening_processor = (
                ctx.video_processor
            )


        @st.fragment(run_every=0.3)
        def monitor_screening():

            processor = (
                st.session_state.get(
                    "screening_processor"
                )
            )

            if processor is None:

                st.markdown(
                    '<div class="checking">'
                    '📷 카메라 준비 중...'
                    '</div>',
                    unsafe_allow_html=True
                )

                return


            elapsed, frame_count, done, frames = (
                processor.get_state()
            )

            progress_ratio = min(1.0, elapsed / SCREENING_SECONDS)

            st.progress(progress_ratio)

            st.caption(
                f"촬영 진행: "
                f"{min(elapsed, SCREENING_SECONDS):0.1f} / "
                f"{SCREENING_SECONDS}초 (프레임 {frame_count}장)"
            )


            if done:

                if not frames:

                    st.error(
                        "촬영된 프레임이 없습니다. 다시 시도해 주세요."
                    )

                    return


                with st.spinner("얼굴 랜드마크 분석 중..."):

                    (
                        annotated_rgb,
                        metrics,
                        face_detected
                    ) = generate_landmark_visualization_and_metrics(
                        frames
                    )


                if not face_detected:

                    st.warning(
                        "마지막 프레임에서 얼굴을 정확히 인식하지 "
                        "못했습니다. 결과의 신뢰도가 낮을 수 있습니다."
                    )

                if not metrics["iris_available"]:

                    st.info(
                        "이 환경에서는 눈동자(iris) 랜드마크가 "
                        "검출되지 않아 시선 이동 지표는 이번 결과에서 "
                        "제외됩니다."
                    )


                st.session_state.screening_result = {
                    "annotated_rgb": annotated_rgb,
                    "metrics": metrics,
                    "face_detected": face_detected,
                }

                st.session_state.stage = "reaction_test"

                st.rerun()


        monitor_screening()


    # ========================================================
    # 분석 결과가 있으면 -> 리포트 표시
    # ========================================================

    else:

        st.info("촬영이 이미 완료되었습니다. 다음 단계로 이동합니다...")
        st.session_state.stage = "reaction_test"
        st.rerun()


# ============================================================
# 반응속도 테스트 단계
# ============================================================

elif st.session_state.stage == "reaction_test":

    st.markdown(
        """
        <div class="pass">
        ⚡ 반응속도 테스트
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "※ 이 테스트는 네트워크·화면 렌더링 지연이 섞여 들어갈 수 "
        "있어 실제 반응속도보다 다소 느리게 측정될 수 있는 "
        "프로토타입 버전입니다."
    )


    if st.session_state.reaction_result is not None:

        st.session_state.stage = "report"
        st.rerun()


    skip_col, _ = st.columns([1, 3])

    with skip_col:

        if st.button("건너뛰기 (측정 안 함)"):

            st.session_state.reaction_result = {
                "reaction_ms": None
            }

            st.session_state.stage = "report"

            st.rerun()


    if st.session_state.reaction_entry_time is None:

        st.session_state.reaction_entry_time = time.time()
        st.session_state.reaction_target_delay = random.uniform(2.0, 5.0)


    @st.fragment(run_every=0.05)
    def run_reaction_test():

        entry_time = st.session_state.reaction_entry_time
        target_delay = st.session_state.reaction_target_delay

        elapsed_since_entry = time.time() - entry_time


        if elapsed_since_entry < target_delay:

            st.markdown(
                """
                <div style="
                    background-color:#333;
                    color:white;
                    text-align:center;
                    padding:60px;
                    border-radius:12px;
                    font-size:20px;
                ">
                🕒 준비하세요...
                </div>
                """,
                unsafe_allow_html=True
            )

            return


        if st.session_state.reaction_stimulus_time is None:

            st.session_state.reaction_stimulus_time = time.time()


        st.markdown(
            """
            <div style="
                background-color:#2ecc71;
                color:white;
                text-align:center;
                padding:60px;
                border-radius:12px;
                font-size:24px;
                font-weight:bold;
            ">
            🟢 지금 클릭하세요!
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("클릭!", use_container_width=True, key="reaction_click_btn"):

            reaction_ms = (
                (time.time() - st.session_state.reaction_stimulus_time)
                * 1000.0
            )

            st.session_state.reaction_result = {
                "reaction_ms": float(reaction_ms)
            }

            st.session_state.stage = "report"

            st.rerun()


    run_reaction_test()


# ============================================================
# 최종 리포트 단계
# ============================================================

elif st.session_state.stage == "report":

    screening = st.session_state.screening_result
    reaction = st.session_state.reaction_result

    if screening is None:

        st.error("촬영 데이터가 없습니다. 처음부터 다시 시도해 주세요.")

    else:

        st.success("촬영 및 분석이 완료되었습니다.")

        st.image(
            screening["annotated_rgb"],
            caption="추출된 얼굴 랜드마크",
            use_container_width=True
        )

        # ========================================================
        # 식별자 입력 (선택) — 입력하면 개인 기준선을 반영해
        # 결과를 계산한다. 입력하지 않아도 결과 확인은 가능하다
        # (이 경우 모집단 기준으로만 계산됨).
        # ========================================================

        st.markdown("### 👤 결과 계산을 위한 식별자 (선택)")

        st.caption(
            "입력하면 이후 측정에서 이번 결과와 비교하고, 반복 "
            "측정이 쌓일수록 이 정보를 바탕으로 결과가 본인에게 "
            "맞춰져요. 입력하지 않아도 결과는 확인할 수 있어요."
        )

        user_id = st.text_input(
            "이름 또는 별명",
            key="baseline_user_id"
        )

        reaction_ms = (reaction or {}).get("reaction_ms")

        risk_score, raw_scores, detail = compute_risk_score(
            screening["metrics"],
            reaction_ms=reaction_ms,
            user_id=(user_id or None)
        )

        # 모집단 통계 갱신 — 같은 세션에서 재실행(rerun)될 때마다
        # 중복 반영되지 않도록 세션당 1회만 수행한다.
        if not st.session_state.population_recorded:
            update_population_stats(screening["metrics"], reaction_ms=reaction_ms)
            st.session_state.population_recorded = True

        st.markdown("### 📋 분석 리포트 (프로토타입)")

        st.caption(
            "※ 이 화면의 결과는 의료기기의 진단이 아니며, 질병의 "
            "유무를 판단하지 않습니다. 각 지표를 살펴보는 방향은 "
            "관련 연구(눈깜빡임률-MCI, 시선패턴-AD, 안면비대칭-AD "
            "등)와 궤를 같이 하지만, 이 화면의 구체적 점수·가중치는 "
            "임상적으로 검증되지 않은 프로토타입 계산값입니다."
        )

        TIER_STYLE = {
            "양호":       {"color": "#2E7D32", "emoji": "🟢"},
            "주의":       {"color": "#B8860B", "emoji": "🟡"},
            "확인 권장":  {"color": "#C0392B", "emoji": "🔴"},
        }

        TIER_MESSAGE = {
            "양호": (
                "안심하셔도 좋습니다. 오늘 측정된 패턴은 양호한 "
                "범위이니, 지금처럼 평소 습관을 유지해 보세요."
            ),
            "주의": (
                "지속적인 관찰이 필요합니다. 평소와 다소 차이가 "
                "있는 패턴이 관찰되니, 컨디션이나 수면 등의 영향일 "
                "수 있는 만큼 다음 기회에 한 번 더 측정해 보시는 "
                "것을 권합니다."
            ),
            "확인 권장": (
                "가까운 신경과·정신건강의학과 또는 치매안심센터 "
                "방문을 권해 드립니다. 평소와 뚜렷한 차이가 있는 "
                "패턴이 관찰되어, 정확한 확인을 받아보시는 것이 "
                "좋겠습니다."
            ),
        }

        tier = detail["tier"]
        style = TIER_STYLE[tier]

        st.markdown(
            f"<div style='padding:16px;border-radius:10px;"
            f"background-color:{style['color']}1A;"
            f"border:1px solid {style['color']};'>"
            f"<span style='font-size:22px;font-weight:700;"
            f"color:{style['color']};'>{style['emoji']} {tier}</span>"
            f"<div style='margin-top:6px;'>{TIER_MESSAGE[tier]}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.expander("지표별 상세 보기 (참고용)"):

            if detail["mode"] == "bootstrap_legacy":
                st.caption(
                    "⚙️ 현재는 이 앱을 사용한 세션 수가 아직 적어 "
                    "(모집단 기준 최소 표본 수 미만), 지표별 실측 "
                    "평균·표준편차 대신 임시 고정 계수로 계산한 "
                    "결과입니다. 사용 세션이 쌓이면 자동으로 "
                    "통계 기반 방식으로 전환됩니다. 아래 값은 "
                    "원시 측정값에 고정 계수를 곱한 참고 수치입니다."
                )
            else:
                st.caption(
                    "각 값은 '평소·모집단 대비 얼마나 벗어났는지'를 "
                    "나타내는 표준화 지표(z+)이며, 원시 측정값 자체가 "
                    "아닙니다. 0에 가까울수록 평소·다른 사용자들과 "
                    "비슷한 패턴입니다."
                )

            for name, val in raw_scores.items():

                if val is None:
                    st.write(f"- {name}: 측정 안 됨 (또는 데이터 축적 중)")
                else:
                    st.write(f"- {name}: {val:.2f}")

            if detail["mode"] == "data_driven":
                if detail["n_personal"] > 0:
                    st.caption(
                        f"※ 개인 기록 {detail['n_personal']}회가 반영된 "
                        f"결과입니다 (기록이 쌓일수록 본인 평소 패턴 "
                        f"비중이 커집니다)."
                    )
                else:
                    st.caption("※ 개인 기록이 없어 모집단 기준으로만 계산됐습니다.")

            st.caption(
                "지표별 누적 세션 수(참고): " + ", ".join(
                    f"{METRIC_LABELS[k]} {v}회"
                    for k, v in detail["population_n"].items()
                )
            )

            st.caption(
                f"내부 계산용 참고 수치(비공개 스코어): {risk_score:.1f} / 100 "
                "— 화면에 노출되는 최종 결과는 위 3단계 판정이며, 이 "
                "수치 자체는 정상·비정상을 가르는 절대 기준이 아닙니다."
            )

        # NOTE: 이전에는 여기에 st.progress()와 "참고용 종합 점수
        # N/100" 캡션을 화면에 노출했으나, 숫자 스코어가 "위험도"로
        # 오독될 소지가 있어 제거했다. 결과 화면의 주 정보는 위의
        # 3단계 판정(양호/주의/확인 권장)이며, 원시 참고 수치는
        # "지표별 상세 보기" expander 안에서만 확인할 수 있다.


        # ========================================================
        # 개인 기준선 비교
        # ========================================================

        st.markdown("---")
        st.markdown("### 📈 이전 측정치와 비교")

        st.caption(
            "※ '정상인 평균'이 아니라, 같은 사람의 과거 측정치와 "
            "비교하는 기능입니다. 절대적인 정상 범위를 판단하는 "
            "것이 아니라, 본인의 변화 추이만을 보여줍니다."
        )

        if user_id:

            baseline = get_baseline(user_id)

            if baseline is None:

                if st.button("이번 결과를 내 기준선으로 저장", use_container_width=True):

                    save_baseline(
                        user_id, screening["metrics"], risk_score,
                        reaction_ms=reaction_ms
                    )

                    st.success(
                        f"'{user_id}'님의 기준선으로 저장했어요. "
                        f"다음 검사부터 이 결과와 비교해서 보여드릴게요."
                    )

            else:

                st.write(
                    f"**기준선 측정일**: {baseline['saved_at']} "
                    f"(당시 참고 점수: {baseline['risk_score']:.1f})"
                )

                base_metrics = baseline["metrics"]
                curr_metrics = screening["metrics"]

                compare_rows = [
                    ("표정 변화", "expression_change"),
                    ("미세 근육 움직임", "micro_movement"),
                    ("안면 비대칭", "facial_asymmetry"),
                    ("눈 깜빡임 빈도(회/분)", "blink_rate"),
                    ("시선 이동 변동성", "gaze_variability"),
                ]

                for label, key in compare_rows:

                    base_val = base_metrics.get(key)
                    curr_val = curr_metrics.get(key)

                    if base_val is None or curr_val is None:
                        st.write(f"- {label}: 비교 불가 (측정 안 됨)")
                        continue

                    delta = curr_val - base_val
                    arrow = "🔺" if delta > 0 else ("🔻" if delta < 0 else "➖")

                    st.write(
                        f"- {label}: 기준 {base_val:.2f} → 현재 "
                        f"{curr_val:.2f} ({arrow} {delta:+.2f})"
                    )

                risk_delta = risk_score - baseline["risk_score"]
                st.write(
                    f"- **참고 점수**: 기준 "
                    f"{baseline['risk_score']:.1f} → 현재 {risk_score:.1f} "
                    f"({risk_delta:+.1f})"
                )

                if st.button(
                    "이번 결과로 기준선 갱신",
                    use_container_width=True
                ):

                    save_baseline(
                        user_id, screening["metrics"], risk_score,
                        reaction_ms=reaction_ms
                    )

                    st.success("기준선을 이번 결과로 갱신했어요.")

        else:
            st.info("위에 이름 또는 별명을 입력하면 이 기능을 사용할 수 있어요.")


        if st.button(
            "🔄 처음부터 다시 검사",
            use_container_width=True
        ):

            reset_app()

            st.rerun()


# ============================================================
# 안내
# ============================================================

st.markdown("---")

st.caption(
    "※ 본 프로그램은 촬영 전 사전 조건 확인용 프로토타입이며 "
    "의료적 진단을 대신하지 않습니다."
)
