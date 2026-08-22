import streamlit as st
import cv2
import numpy as np
from collections import deque

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

from av import VideoFrame


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="안경 착용 엄격 검사",
    page_icon="🔍",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .guide {
        padding: 15px;
        border-radius: 12px;
        background: #eef6ff;
        margin-bottom: 15px;
        font-size: 16px;
    }

    .ok-box {
        padding: 18px;
        border-radius: 12px;
        background: #e9fff0;
        border: 2px solid #00c853;
        color: #008a3e;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
    }

    .bad-box {
        padding: 18px;
        border-radius: 12px;
        background: #fff0f0;
        border: 2px solid #ff3030;
        color: #d60000;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
    }

    .wait-box {
        padding: 18px;
        border-radius: 12px;
        background: #fff9e6;
        border: 2px solid #ffb300;
        color: #a86b00;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="main-title">🔍 안경 착용 엄격 검사</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="guide">
    촬영 전 <b>안경·선글라스·마스크·모자</b>를 벗고
    정면을 바라봐 주세요.<br><br>
    안경 착용이 확인되거나 판정이 불확실하면
    <b>촬영을 허용하지 않습니다.</b>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OpenCV 확인
# ============================================================

if not hasattr(cv2, "CascadeClassifier"):

    st.error("❌ OpenCV가 정상적으로 로딩되지 않았습니다.")

    st.code(
        f"""
OpenCV version:
{getattr(cv2, "__version__", "unknown")}

OpenCV path:
{getattr(cv2, "__file__", "unknown")}
"""
    )

    st.stop()


# ============================================================
# Haar Cascade
# ============================================================

CASCADE_DIR = cv2.data.haarcascades

FACE_XML = (
    CASCADE_DIR +
    "haarcascade_frontalface_default.xml"
)

EYE_XML = (
    CASCADE_DIR +
    "haarcascade_eye.xml"
)

GLASSES_XML = (
    CASCADE_DIR +
    "haarcascade_eye_tree_eyeglasses.xml"
)


face_cascade = cv2.CascadeClassifier(FACE_XML)
eye_cascade = cv2.CascadeClassifier(EYE_XML)
glasses_cascade = cv2.CascadeClassifier(GLASSES_XML)


if face_cascade.empty():
    st.error("❌ 얼굴 검출 모델을 불러오지 못했습니다.")
    st.stop()


if eye_cascade.empty():
    st.error("❌ 눈 검출 모델을 불러오지 못했습니다.")
    st.stop()


# ============================================================
# 안경 검출 함수
# ============================================================

def detect_glasses(frame):

    result = {
        "status": "UNKNOWN",
        "reason": "판정 대기",
        "face": None,
        "eyes": 0,
        "glasses": 0,
        "edge_score": 0.0,
        "line_score": 0,
    }

    if frame is None:
        return result


    # --------------------------------------------------------
    # Gray
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # 얼굴 찾기
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.10,
        minNeighbors=5,
        minSize=(120, 120),
    )


    if len(faces) == 0:

        result["reason"] = "얼굴을 찾지 못했습니다."

        return result


    # 가장 큰 얼굴
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    result["face"] = (
        int(x),
        int(y),
        int(w),
        int(h)
    )


    # --------------------------------------------------------
    # 얼굴이 너무 작으면 검사하지 않음
    # --------------------------------------------------------

    if w < 170 or h < 170:

        result["reason"] = "얼굴을 조금 더 가까이 해주세요."

        return result


    # --------------------------------------------------------
    # 눈/안경 검사 영역
    # --------------------------------------------------------

    ex1 = max(
        0,
        int(x + 0.05 * w)
    )

    ex2 = min(
        gray.shape[1],
        int(x + 0.95 * w)
    )

    ey1 = max(
        0,
        int(y + 0.16 * h)
    )

    ey2 = min(
        gray.shape[0],
        int(y + 0.58 * h)
    )


    eye_region = gray[
        ey1:ey2,
        ex1:ex2
    ]


    if eye_region.size == 0:

        result["reason"] = "눈 영역을 만들 수 없습니다."

        return result


    # ========================================================
    # 눈 검출
    # ========================================================

    try:

        eyes = eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.06,
            minNeighbors=4,
            minSize=(22, 18),
        )

        eye_count = len(eyes)

    except Exception:

        eye_count = 0


    result["eyes"] = eye_count


    # ========================================================
    # 안경 Cascade
    #
    # 중요:
    # 이것은 단독 판정에 사용하지 않는다.
    # ========================================================

    glasses_count = 0

    try:

        if not glasses_cascade.empty():

            glasses = glasses_cascade.detectMultiScale(
                eye_region,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(25, 18),
            )

            glasses_count = len(glasses)

    except Exception:

        glasses_count = 0


    result["glasses"] = glasses_count


    # ========================================================
    # 안경테 구조 분석
    #
    # 단순 glasses cascade가 아니라
    # 눈 주변의 프레임 형태를 보조적으로 검사한다.
    # ========================================================

    resized = cv2.resize(
        eye_region,
        (320, 180)
    )


    # 조명 차이를 줄이기 위한 CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    normalized = clahe.apply(
        resized
    )


    blurred = cv2.GaussianBlur(
        normalized,
        (5, 5),
        0
    )


    edges = cv2.Canny(
        blurred,
        50,
        150
    )


    # --------------------------------------------------------
    # Edge density
    # --------------------------------------------------------

    edge_score = float(
        np.mean(edges > 0)
    )


    result["edge_score"] = edge_score


    # --------------------------------------------------------
    # 선분 검사
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=25,
        minLineLength=25,
        maxLineGap=5,
    )


    horizontal = 0
    vertical = 0


    if lines is not None:

        for line in lines[:, 0]:

            x1, y1, x2, y2 = line

            angle = abs(
                np.degrees(
                    np.arctan2(
                        y2 - y1,
                        x2 - x1
                    )
                )
            )


            if (
                angle < 12
                or angle > 168
            ):

                horizontal += 1


            if (
                78 < angle < 102
            ):

                vertical += 1


    line_score = (
        horizontal +
        vertical
    )


    result["line_score"] = line_score


    # ========================================================
    # 핵심 판정
    # ========================================================

    #
    # 1) 강한 안경 프레임 증거
    #
    # 현재 테스트 사진의 맨눈 오탐을 줄이기 위해
    # edge + line을 함께 요구한다.
    #

    strong_frame = (
        edge_score >= 0.055
        and
        line_score >= 8
    )


    #
    # 2) Cascade + 프레임 구조가 동시에 의심되는 경우
    #

    cascade_frame = (
        glasses_count >= 2
        and
        edge_score >= 0.045
    )


    #
    # 3) 최종 안경 의심
    #

    if strong_frame or cascade_frame:

        result["status"] = "GLASSES"

        result["reason"] = (
            "안경테 구조가 반복적으로 의심됩니다."
        )

        return result


    # ========================================================
    # 맨눈 후보
    # ========================================================

    if eye_count >= 2:

        result["status"] = "NO_GLASSES"

        result["reason"] = (
            "양쪽 눈이 확인되었습니다."
        )

        return result


    # ========================================================
    # 불확실
    # ========================================================

    result["status"] = "UNKNOWN"

    result["reason"] = (
        "눈 또는 안경 상태를 확실하게 판단할 수 없습니다."
    )

    return result


# ============================================================
# WebRTC Processor
# ============================================================

class GlassesProcessor(VideoProcessorBase):

    def __init__(self):

        self.history = deque(
            maxlen=12
        )

        self.last_result = None

        self.frame_count = 0


    def recv(self, frame):

        try:

            img = frame.to_ndarray(
                format="bgr24"
            )

        except Exception:

            return frame


        # ----------------------------------------------------
        # 분석
        # ----------------------------------------------------

        result = detect_glasses(
            img
        )


        self.last_result = result

        self.frame_count += 1


        # 최근 판정 저장
        self.history.append(
            result["status"]
        )


        # ----------------------------------------------------
        # 최종 표시 상태
        # ----------------------------------------------------

        final_status = self.get_final_status()


        h, w = img.shape[:2]


        # ----------------------------------------------------
        # 얼굴 박스
        # ----------------------------------------------------

        face = result.get(
            "face"
        )


        if face is not None:

            x, y, fw, fh = face

        else:

            x = int(w * 0.20)
            y = int(h * 0.10)
            fw = int(w * 0.60)
            fh = int(h * 0.80)


        # ====================================================
        # 상태별 색상
        # ====================================================

        if final_status == "GLASSES":

            color = (
                0,
                0,
                255
            )

            message = (
                "NOT ELIGIBLE - GLASSES"
            )


        elif final_status == "NO_GLASSES":

            color = (
                0,
                220,
                0
            )

            message = (
                "ELIGIBLE - NO GLASSES"
            )


        else:

            color = (
                0,
                180,
                255
            )

            message = (
                "CHECKING - DO NOT SHOOT"
            )


        # ----------------------------------------------------
        # 얼굴 가이드
        # ----------------------------------------------------

        cv2.rectangle(
            img,
            (x, y),
            (x + fw, y + fh),
            color,
            3
        )


        # ----------------------------------------------------
        # 상단 상태
        # ----------------------------------------------------

        cv2.rectangle(
            img,
            (10, 10),
            (min(w - 10, 560), 55),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            img,
            message,
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA
        )


        return VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


    # ========================================================
    # 최근 프레임 종합 판정
    # ========================================================

    def get_final_status(self):

        if len(self.history) < 6:

            return "UNKNOWN"


        history = list(
            self.history
        )


        glasses = history.count(
            "GLASSES"
        )

        no_glasses = history.count(
            "NO_GLASSES"
        )

        unknown = history.count(
            "UNKNOWN"
        )


        # ====================================================
        # 1. 안경
        #
        # 최근 12프레임 중 3프레임 이상
        # 강한 안경 증거가 있으면 부적합
        # ====================================================

        if glasses >= 3:

            return "GLASSES"


        # ====================================================
        # 2. 불확실
        #
        # 애매한 프레임이 너무 많으면 촬영 금지
        # ====================================================

        if unknown >= 5:

            return "UNKNOWN"


        # ====================================================
        # 3. 맨눈
        #
        # 최소 6프레임 이상 맨눈 확인
        # ====================================================

        if no_glasses >= 6:

            return "NO_GLASSES"


        return "UNKNOWN"


# ============================================================
# Session State
# ============================================================

if "capture_allowed" not in st.session_state:

    st.session_state.capture_allowed = False


# ============================================================
# 카메라
# ============================================================

st.subheader(
    "📷 카메라 검사"
)


st.write(
    "얼굴을 화면 중앙에 맞추고 "
    "잠시 정면을 바라봐 주세요."
)


st.caption(
    "안경 착용 → 촬영 불가"
)


st.caption(
    "판정 불확실 → 촬영 불가"
)


st.caption(
    "맨눈 확인 → 촬영 가능"
)


# ============================================================
# WebRTC
# ============================================================

ctx = webrtc_streamer(

    key="strict-glasses-check-v3",

    mode=WebRtcMode.SENDRECV,

    video_processor_factory=GlassesProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False,
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
    },
)


# ============================================================
# 판정 결과
# ============================================================

if ctx.video_processor:

    processor = (
        ctx.video_processor
    )


    final_status = (
        processor.get_final_status()
    )


    result = (
        processor.last_result
    )


    # --------------------------------------------------------
    # 1. 맨눈
    # --------------------------------------------------------

    if final_status == "NO_GLASSES":

        st.markdown(
            """
            <div class="ok-box">
            ✅ 촬영 가능합니다
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.success(
            "안경이 확인되지 않았습니다. "
            "촬영 조건을 충족했습니다."
        )


        st.session_state.capture_allowed = True


    # --------------------------------------------------------
    # 2. 안경
    # --------------------------------------------------------

    elif final_status == "GLASSES":

        st.markdown(
            """
            <div class="bad-box">
            ❌ 부적합 — 안경 착용
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.error(
            "안경 착용이 의심됩니다. "
            "안경을 벗고 다시 검사해주세요."
        )


        st.session_state.capture_allowed = False


    # --------------------------------------------------------
    # 3. 불확실
    # --------------------------------------------------------

    else:

        st.markdown(
            """
            <div class="wait-box">
            ⚠️ 검사 중 — 촬영 불가
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.warning(
            "판정이 아직 확실하지 않습니다. "
            "정면을 바라보고 잠시 기다려주세요."
        )


        st.session_state.capture_allowed = False


    # ========================================================
    # 검사 정보
    # ========================================================

    if result is not None:

        st.caption(
            f"검사 프레임: {processor.frame_count}"
        )

        st.caption(
            f"최근 눈 검출: {result.get('eyes', 0)}"
        )

        st.caption(
            f"최근 안경 Cascade: {result.get('glasses', 0)}"
        )

        st.caption(
            f"프레임 구조 점수: "
            f"{result.get('edge_score', 0):.3f}"
        )


# ============================================================
# 촬영 버튼
# ============================================================

st.markdown("---")


if st.session_state.capture_allowed:

    if st.button(
        "📸 촬영하기",
        type="primary",
        use_container_width=True,
    ):

        st.success(
            "📸 촬영을 시작할 수 있습니다."
        )

else:

    st.button(
        "🔒 촬영 불가",
        disabled=True,
        use_container_width=True,
    )


# ============================================================
# 안내
# ============================================================

st.markdown("---")

st.caption(
    "※ 본 검사는 OpenCV 기반 비접촉 사전검사입니다. "
    "조명·카메라·얼굴 각도에 따라 오판정이 발생할 수 있으며, "
    "의료적 진단이나 공식 신원확인 용도로 사용해서는 안 됩니다."
)
