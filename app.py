import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from av import VideoFrame


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI 비접촉 선별검사 시스템",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:32px;
    font-weight:800;
    margin-top:10px;
    margin-bottom:8px;
}

.subtitle {
    text-align:center;
    color:#666;
    margin-bottom:20px;
}

.pass-box {
    padding:18px;
    border-radius:14px;
    border:3px solid #00c853;
    background:#ecfff3;
    color:#00883e;
    text-align:center;
    font-size:24px;
    font-weight:800;
}

.fail-box {
    padding:18px;
    border-radius:14px;
    border:3px solid #ff3030;
    background:#fff0f0;
    color:#d00000;
    text-align:center;
    font-size:24px;
    font-weight:800;
}

.wait-box {
    padding:18px;
    border-radius:14px;
    border:3px solid #ffb300;
    background:#fff8e1;
    color:#9b6500;
    text-align:center;
    font-size:22px;
    font-weight:800;
}

.stage-box {
    padding:20px;
    border-radius:14px;
    background:#eef6ff;
    border:2px solid #4285f4;
    text-align:center;
    font-size:22px;
    font-weight:800;
}

.small {
    font-size:14px;
    color:#777;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 AI 비접촉 선별검사 시스템</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">촬영 전 얼굴 상태 및 촬영 조건 자동 검사</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "stage" not in st.session_state:
    st.session_state.stage = "precheck"

if "started_at" not in st.session_state:
    st.session_state.started_at = time.time()

if "auto_started" not in st.session_state:
    st.session_state.auto_started = False


# ============================================================
# RESET
# ============================================================

def reset_app():

    st.session_state.stage = "precheck"
    st.session_state.auto_started = False
    st.session_state.started_at = time.time()


# ============================================================
# CASCADE
# ============================================================

FACE_XML = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"

EYE_XML = cv2.data.haarcascades + \
    "haarcascade_eye.xml"

GLASSES_XML = cv2.data.haarcascades + \
    "haarcascade_eye_tree_eyeglasses.xml"


face_cascade = cv2.CascadeClassifier(FACE_XML)

eye_cascade = cv2.CascadeClassifier(EYE_XML)

glasses_cascade = cv2.CascadeClassifier(GLASSES_XML)


if face_cascade.empty():

    st.error("얼굴 검출 모델을 불러오지 못했습니다.")

    st.stop()


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_frame(frame):

    result = {
        "face": False,
        "eyes": False,
        "glasses": False,
        "mask": False,
        "hat": False,
        "frontal": False,
        "center": False,
        "distance": False,
        "lighting": False,
        "status": "UNKNOWN",
        "message": "검사 중"
    }

    if frame is None:
        return result


    h, w = frame.shape[:2]


    # --------------------------------------------------------
    # GRAY
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # 밝기
    # --------------------------------------------------------

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )


    if (
        brightness >= 45
        and brightness <= 220
        and contrast >= 20
    ):

        result["lighting"] = True


    # --------------------------------------------------------
    # 얼굴
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )


    if len(faces) == 0:

        result["message"] = \
            "얼굴을 찾을 수 없습니다."

        return result


    # 가장 큰 얼굴
    x, y, fw, fh = max(
        faces,
        key=lambda a: a[2] * a[3]
    )


    result["face"] = True


    # --------------------------------------------------------
    # 얼굴 크기
    # --------------------------------------------------------

    face_ratio = fw / float(w)

    if 0.25 <= face_ratio <= 0.75:
        result["distance"] = True


    # --------------------------------------------------------
    # 중앙
    # --------------------------------------------------------

    face_cx = x + fw / 2
    frame_cx = w / 2

    center_error = abs(
        face_cx - frame_cx
    ) / w


    if center_error <= 0.15:
        result["center"] = True


    # --------------------------------------------------------
    # 얼굴 ROI
    # --------------------------------------------------------

    roi_x1 = max(0, x)
    roi_y1 = max(0, y)
    roi_x2 = min(w, x + fw)
    roi_y2 = min(h, y + fh)

    face_roi = gray[
        roi_y1:roi_y2,
        roi_x1:roi_x2
    ]


    if face_roi.size == 0:
        return result


    # --------------------------------------------------------
    # 눈 영역
    # --------------------------------------------------------

    eye_y1 = int(fh * 0.18)
    eye_y2 = int(fh * 0.58)

    eye_region = face_roi[
        eye_y1:eye_y2,
        :
    ]


    if eye_region.size > 0:

        eyes = eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.06,
            minNeighbors=4,
            minSize=(20, 18)
        )

        if len(eyes) >= 2:

            result["eyes"] = True


    # --------------------------------------------------------
    # 안경 검사
    #
    # 단일 cascade 검출만으로 부적합 판정하지 않는다.
    # --------------------------------------------------------

    glasses_hits = 0

    if not glasses_cascade.empty():

        try:

            glasses = glasses_cascade.detectMultiScale(
                eye_region,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 20)
            )

            glasses_hits = len(glasses)

        except Exception:

            glasses_hits = 0


    # --------------------------------------------------------
    # 눈 주변 Edge
    # --------------------------------------------------------

    if eye_region.size > 0:

        eye_resize = cv2.resize(
            eye_region,
            (320, 150)
        )

        eye_blur = cv2.GaussianBlur(
            eye_resize,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            eye_blur,
            60,
            150
        )

        edge_ratio = float(
            np.mean(edges > 0)
        )

    else:

        edge_ratio = 0


    # --------------------------------------------------------
    # 안경 프레임 구조
    # --------------------------------------------------------

    lines = None

    if eye_region.size > 0:

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=28,
            minLineLength=35,
            maxLineGap=6
        )


    horizontal = 0

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

            if angle < 12 or angle > 168:

                horizontal += 1


    # ========================================================
    # 안경 최종 판단
    #
    # 보수적으로 판단
    # ========================================================

    glasses_structure = (
        glasses_hits >= 2
        and
        edge_ratio >= 0.045
        and
        horizontal >= 3
    )


    if glasses_structure:

        result["glasses"] = True


    # ========================================================
    # 마스크 추정
    #
    # 아래 얼굴 영역이 비정상적으로 균일한 경우
    # 마스크 가능성을 높인다.
    # ========================================================

    lower_y1 = int(fh * 0.48)
    lower_y2 = int(fh * 0.92)

    lower_face = face_roi[
        lower_y1:lower_y2,
        :
    ]


    if lower_face.size > 0:

        lower_std = float(
            np.std(lower_face)
        )

        lower_mean = float(
            np.mean(lower_face)
        )

        # 얼굴 하단이 지나치게 균일하면서
        # 밝기 차이가 큰 경우 마스크 의심
        if (
            lower_std < 28
            and
            60 < lower_mean < 220
        ):

            # 너무 공격적으로 잡지 않기 위해
            # 단독으로 바로 부적합하지 않는다.
            mask_candidate = True

        else:

            mask_candidate = False

    else:

        mask_candidate = False


    # ========================================================
    # 모자 추정
    #
    # 얼굴 위쪽 영역을 분석
    # ========================================================

    upper_y1 = 0
    upper_y2 = int(fh * 0.25)

    upper_face = face_roi[
        upper_y1:upper_y2,
        :
    ]


    if upper_face.size > 0:

        upper_edges = cv2.Canny(
            upper_face,
            50,
            140
        )

        upper_edge_ratio = float(
            np.mean(upper_edges > 0)
        )

    else:

        upper_edge_ratio = 0


    # --------------------------------------------------------
    # 모자 후보
    # --------------------------------------------------------

    hat_candidate = (
        upper_edge_ratio > 0.18
    )


    # ========================================================
    # 정면 검사
    # ========================================================

    frontal = False

    if len(eyes) >= 2:

        centers = []

        for ex, ey, ew, eh in eyes:

            centers.append(
                (
                    ex + ew / 2,
                    ey + eh / 2
                )
            )


        if len(centers) >= 2:

            centers = sorted(
                centers,
                key=lambda p: p[0]
            )

            left = centers[0]
            right = centers[-1]

            eye_distance = abs(
                right[0] - left[0]
            )

            eye_y_difference = abs(
                right[1] - left[1]
            )

            if (
                eye_distance > 30
                and
                eye_y_difference <
                eye_distance * 0.35
            ):

                frontal = True


    result["frontal"] = frontal


    # ========================================================
    # 마스크/모자 최종 판정
    #
    # 휴리스틱은 단독 판정으로 사용하지 않고
    # 다른 품질 조건과 결합
    # ========================================================

    if mask_candidate and not result["eyes"]:

        result["mask"] = True


    if hat_candidate:

        # 위쪽 영역이 매우 강하게 가려진 경우만
        # 모자 후보
        if upper_edge_ratio > 0.24:

            result["hat"] = True


    # ========================================================
    # 최종 상태
    # ========================================================

    if result["glasses"]:

        result["status"] = "FAIL"

        result["message"] = \
            "안경 또는 선글라스가 감지되었습니다."

        return result


    if result["mask"]:

        result["status"] = "FAIL"

        result["message"] = \
            "마스크 착용이 의심됩니다."

        return result


    if result["hat"]:

        result["status"] = "FAIL"

        result["message"] = \
            "모자 착용이 의심됩니다."

        return result


    # --------------------------------------------------------
    # 필수 촬영 조건
    # --------------------------------------------------------

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

        result["message"] = \
            "촬영 조건이 충족되었습니다."

        return result


    result["status"] = "UNKNOWN"

    if not result["face"]:

        result["message"] = \
            "얼굴을 정면으로 보여주세요."

    elif not result["eyes"]:

        result["message"] = \
            "양쪽 눈이 보이도록 해주세요."

    elif not result["center"]:

        result["message"] = \
            "얼굴을 화면 중앙에 맞춰주세요."

    elif not result["distance"]:

        result["message"] = \
            "카메라와 거리를 조절해주세요."

    elif not result["lighting"]:

        result["message"] = \
            "조명을 밝게 해주세요."

    elif not result["frontal"]:

        result["message"] = \
            "정면을 바라봐 주세요."

    return result


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class PrecheckProcessor(VideoProcessorBase):

    def __init__(self):

        self.history = deque(
            maxlen=12
        )

        self.latest = None

        self.pass_count = 0

        self.fail_count = 0

        self.frame_count = 0


    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        result = analyze_frame(
            img
        )

        self.latest = result

        self.frame_count += 1

        self.history.append(
            result["status"]
        )


        # ----------------------------------------------------
        # 연속 PASS
        # ----------------------------------------------------

        if result["status"] == "PASS":

            self.pass_count += 1
            self.fail_count = 0

        elif result["status"] == "FAIL":

            self.fail_count += 1
            self.pass_count = 0

        else:

            self.pass_count = 0


        # ----------------------------------------------------
        # 상태
        # ----------------------------------------------------

        final_status = self.final_status()


        # ----------------------------------------------------
        # 색상
        # ----------------------------------------------------

        if final_status == "PASS":

            color = (
                0,
                220,
                0
            )

            text = "READY - CAPTURE CONDITIONS OK"


        elif final_status == "FAIL":

            color = (
                0,
                0,
                255
            )

            text = "NOT ELIGIBLE"


        else:

            color = (
                0,
                180,
                255
            )

            text = "CHECKING - DO NOT SHOOT"


        # ----------------------------------------------------
        # 얼굴 박스
        # ----------------------------------------------------

        face = result.get(
            "face"
        )


        # 얼굴 좌표를 다시 찾음
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)
        )


        if len(faces) > 0:

            x, y, fw, fh = max(
                faces,
                key=lambda a: a[2] * a[3]
            )

            cv2.rectangle(
                img,
                (x, y),
                (x + fw, y + fh),
                color,
                3
            )


        # ----------------------------------------------------
        # 상태 표시
        # ----------------------------------------------------

        cv2.rectangle(
            img,
            (10, 10),
            (min(img.shape[1] - 10, 600), 65),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            img,
            text,
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA
        )


        return VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


    # ========================================================
    # 최종 상태
    # ========================================================

    def final_status(self):

        if self.fail_count >= 3:

            return "FAIL"


        if self.pass_count >= 8:

            return "PASS"


        return "UNKNOWN"


# ============================================================
# PRECHECK SCREEN
# ============================================================

if st.session_state.stage == "precheck":

    st.markdown(
        """
        <div class="stage-box">
        📷 촬영 조건 자동 검사
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    st.info(
        "안경·선글라스·마스크·모자 등을 제거하고 "
        "정면을 바라봐 주세요."
    )


    st.caption(
        "모든 조건이 일정 시간 연속 충족되면 "
        "촬영 버튼 없이 자동으로 선별검사 단계로 이동합니다."
    )


    ctx = webrtc_streamer(

        key="precheck-camera-v5",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=PrecheckProcessor,

        media_stream_constraints={
            "video": True,
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


    # ========================================================
    # STATUS
    # ========================================================

    if ctx.video_processor:

        processor = ctx.video_processor

        final_status = processor.final_status()

        result = processor.latest


        # ----------------------------------------------------
        # PASS
        # ----------------------------------------------------

        if final_status == "PASS":

            st.markdown(
                """
                <div class="pass-box">
                ✅ 촬영 조건 완료
                </div>
                """,
                unsafe_allow_html=True
            )


            st.success(
                "촬영 조건이 충족되었습니다. "
                "선별검사를 시작합니다."
            )


            # 자동 전환
            if not st.session_state.auto_started:

                st.session_state.auto_started = True

                time.sleep(1.2)

                st.session_state.stage = \
                    "screening"

                st.rerun()


        # ----------------------------------------------------
        # FAIL
        # ----------------------------------------------------

        elif final_status == "FAIL":

            st.markdown(
                """
                <div class="fail-box">
                ❌ 부적합 — 촬영 불가
                </div>
                """,
                unsafe_allow_html=True
            )


            if result:

                st.error(
                    result.get(
                        "message",
                        "촬영 조건을 충족하지 못했습니다."
                    )
                )


            st.session_state.auto_started = False


        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        else:

            st.markdown(
                """
                <div class="wait-box">
                🔎 검사 중 — 촬영 불가
                </div>
                """,
                unsafe_allow_html=True
            )


            if result:

                st.warning(
                    result.get(
                        "message",
                        "촬영 조건을 확인하고 있습니다."
                    )
                )


        # ----------------------------------------------------
        # 검사 상태
        # ----------------------------------------------------

        st.caption(
            f"검사 프레임: {processor.frame_count}"
        )

        st.caption(
            f"연속 통과: {processor.pass_count}/8"
        )

        st.caption(
            "안전한 촬영을 위해 판정이 불확실하면 "
            "자동으로 촬영을 차단합니다."
        )


# ============================================================
# SCREENING
# ============================================================

elif st.session_state.stage == "screening":

    st.markdown(
        """
        <div class="stage-box">
        🧠 선별검사 시작
        </div>
        """,
        unsafe_allow_html=True
    )


    st.success(
        "촬영 조건 검사가 완료되었습니다."
    )


    st.write("")

    st.markdown(
        """
        ### 🔎 선별검사 진행

        촬영 조건이 충족되어 다음 단계로 이동했습니다.

        **선별검사 준비가 완료되었습니다.**
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # 현재 단계
    # --------------------------------------------------------

    progress = st.progress(
        0
    )


    status = st.empty()


    for i in range(101):

        progress.progress(i)

        if i < 30:

            status.info(
                "① 얼굴 영상 품질 확인 중..."
            )

        elif i < 60:

            status.info(
                "② 얼굴 특징 데이터 준비 중..."
            )

        elif i < 85:

            status.info(
                "③ 선별검사 분석 준비 중..."
            )

        else:

            status.success(
                "④ 선별검사 분석 준비 완료"
            )

        time.sleep(0.02)


    st.markdown("---")


    st.markdown(
        """
        <div class="pass-box">
        🧠 선별검사 단계 진입 완료
        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        "이 화면은 선별검사 엔진을 연결하기 위한 단계입니다. "
        "실제 의료 AI 분석 모델은 별도의 검증된 모델을 연결해야 합니다."
    )


    if st.button(
        "🔄 처음부터 다시 검사",
        use_container_width=True
    ):

        reset_app()

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="small">
    ※ 본 프로그램의 촬영 조건 검사는 비접촉 사전검사용이며
    의료적 진단을 대신하지 않습니다.
    </div>
    """,
    unsafe_allow_html=True
)
