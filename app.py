import streamlit as st
import cv2
import numpy as np
import threading
import time
import os
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


# ============================================================
# 초기화
# ============================================================

def reset_app():

    st.session_state.stage = "precheck"
    st.session_state.processor = None
    st.session_state.transitioning = False
    st.session_state.screening_processor = None
    st.session_state.screening_result = None


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

def detect_glasses_raw(eye_region, eyes):

    if eye_region.size == 0 or glasses_cascade.empty():
        return False

    if eyes is None or len(eyes) < 2:
        return False

    eh_region, ew_region = eye_region.shape[:2]

    # 검출된 눈 중 좌/우 대표 2개만 사용
    eyes_sorted = sorted(eyes, key=lambda b: b[0])
    left_eye = eyes_sorted[0]
    right_eye = eyes_sorted[-1]

    votes = 0

    for (ex, ey, ew, eh) in (left_eye, right_eye):

        # 눈 박스를 기준으로 좌우/상하로 패딩을 준 패치 생성
        pad_x = int(ew * 0.6)
        pad_y = int(eh * 0.8)

        x1 = max(0, ex - pad_x)
        x2 = min(ew_region, ex + ew + pad_x)
        y1 = max(0, ey - pad_y)
        y2 = min(eh_region, ey + eh + pad_y)

        patch = eye_region[y1:y2, x1:x2]

        if patch.size == 0:
            continue

        try:
            glasses_candidates = glasses_cascade.detectMultiScale(
                patch,
                scaleFactor=1.05,
                minNeighbors=10,
                minSize=(30, 22)
            )
            glasses_count = len(glasses_candidates)
        except Exception:
            glasses_count = 0

        resized_patch = cv2.resize(patch, (160, 120))
        blurred_patch = cv2.GaussianBlur(resized_patch, (5, 5), 0)
        edges = cv2.Canny(blurred_patch, 70, 160)

        edge_ratio = float(np.mean(edges > 0))

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=35,
            maxLineGap=4
        )

        horizontal_lines = 0

        if lines is not None:
            for line in lines[:, 0]:
                x1l, y1l, x2l, y2l = line
                angle = abs(np.degrees(np.arctan2(y2l - y1l, x2l - x1l)))
                if angle < 10 or angle > 170:
                    horizontal_lines += 1

        # 눈 하나(패치)마다 강한 조건을 동시에 요구
        patch_glasses = (
            glasses_count >= 1
            and horizontal_lines >= 3
            and edge_ratio >= 0.06
        )

        if patch_glasses:
            votes += 1

    # 양쪽 눈 패치 중 최소 한쪽에서 강하게 검출되어야
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

def generate_landmark_visualization_and_metrics(frames):

    landmarker = get_face_landmarker()

    target_frame = frames[len(frames) // 2].copy()
    h, w = target_frame.shape[:2]

    motion_deltas = []
    prev_pts = None

    # 프레임이 많을 수 있으니 최대 15프레임 정도만 샘플링해서
    # motion(움직임) 지표를 계산한다 (전체를 다 돌리면 느려짐)
    step = max(1, len(frames) // 15)

    for idx in range(0, len(frames), step):

        f_rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)
        res = landmarker.detect(mp_image)

        if res.face_landmarks:

            curr_pts = np.array(
                [(lm.x, lm.y) for lm in res.face_landmarks[0]]
            )

            if prev_pts is not None and prev_pts.shape == curr_pts.shape:
                motion_deltas.append(
                    float(np.mean(np.abs(curr_pts - prev_pts)))
                )

            prev_pts = curr_pts

    # 중간 프레임에 랜드마크를 찍어 시각화용 이미지 생성
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

    avg_motion = float(np.mean(motion_deltas) * 1000) if motion_deltas else 0.0
    dynamic_stability = float(np.std(motion_deltas) * 500) if motion_deltas else 0.0

    annotated_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

    return annotated_rgb, avg_motion, dynamic_stability, face_detected


def compute_risk_score(avg_motion, dynamic_stability):

    # NOTE: 아래 가중치는 임상적으로 검증된 값이 아닌 프로토타입용
    # 임의 스케일링 값이다. 실사용 전 반드시 검증이 필요하다.
    raw_scores = {
        "머리·표정 움직임 지표": avg_motion * 0.6,
        "동적 안정성 지표": dynamic_stability * 0.8,
    }

    risk_score = float(
        np.clip(np.mean(list(raw_scores.values())) * 2.2, 0.0, 100.0)
    )

    return risk_score, raw_scores


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

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
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

            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
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
                        avg_motion,
                        dynamic_stability,
                        face_detected
                    ) = generate_landmark_visualization_and_metrics(
                        frames
                    )


                if not face_detected:

                    st.warning(
                        "마지막 프레임에서 얼굴을 정확히 인식하지 "
                        "못했습니다. 결과의 신뢰도가 낮을 수 있습니다."
                    )


                risk_score, raw_scores = compute_risk_score(
                    avg_motion,
                    dynamic_stability
                )

                st.session_state.screening_result = {
                    "annotated_rgb": annotated_rgb,
                    "avg_motion": avg_motion,
                    "dynamic_stability": dynamic_stability,
                    "risk_score": risk_score,
                    "raw_scores": raw_scores,
                    "face_detected": face_detected,
                }

                st.rerun()


        monitor_screening()


    # ========================================================
    # 분석 결과가 있으면 -> 리포트 표시
    # ========================================================

    else:

        result = st.session_state.screening_result

        st.success(
            "촬영 및 분석이 완료되었습니다."
        )

        st.image(
            result["annotated_rgb"],
            caption="추출된 얼굴 랜드마크",
            use_container_width=True
        )


        st.markdown("### 📋 분석 리포트 (프로토타입)")

        st.caption(
            "※ 아래 지표는 검증되지 않은 프로토타입 계산값이며, "
            "의학적 진단 근거로 사용할 수 없습니다."
        )

        for name, val in result["raw_scores"].items():

            st.write(
                f"- {name}: {val:.2f} pts"
            )

        st.metric(
            "종합 위험도 스코어 (프로토타입)",
            f"{result['risk_score']:.1f} / 100"
        )

        st.progress(
            result["risk_score"] / 100
        )


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
