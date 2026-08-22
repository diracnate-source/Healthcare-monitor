import streamlit as st
import cv2
import numpy as np
import threading
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from av import VideoFrame


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


# ============================================================
# 초기화
# ============================================================

def reset_app():

    st.session_state.stage = "precheck"
    st.session_state.processor = None
    st.session_state.transitioning = False


# ============================================================
# OpenCV 모델
# ============================================================

FACE_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_XML = cv2.data.haarcascades + "haarcascade_eye.xml"
GLASSES_XML = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"


face_cascade = cv2.CascadeClassifier(FACE_XML)
eye_cascade = cv2.CascadeClassifier(EYE_XML)
glasses_cascade = cv2.CascadeClassifier(GLASSES_XML)


if face_cascade.empty():
    st.error("얼굴 검출 모델을 불러오지 못했습니다.")
    st.stop()


# ============================================================
# 얼굴 분석
# ============================================================

def analyze_frame(frame):

    result = {
        "status": "UNKNOWN",
        "message": "검사 중",

        "face": False,
        "eyes": False,

        "glasses": False,
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
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(20, 20)
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
    # 안경 검사
    #
    # 핵심 수정:
    #
    # 단순히 glasses cascade가 한 번 검출됐다고
    # 부적합 처리하지 않는다.
    #
    # 여러 조건을 동시에 만족할 때만 안경으로 판단한다.
    # ========================================================

    glasses_detected = False


    if eye_region.size > 0 and not glasses_cascade.empty():

        try:

            glasses_candidates = glasses_cascade.detectMultiScale(
                eye_region,
                scaleFactor=1.05,
                minNeighbors=8,
                minSize=(35, 25)
            )

            glasses_count = len(
                glasses_candidates
            )

        except Exception:

            glasses_count = 0


        # Edge 분석
        resized_eye = cv2.resize(
            eye_region,
            (320, 160)
        )

        blurred_eye = cv2.GaussianBlur(
            resized_eye,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred_eye,
            70,
            160
        )


        edge_ratio = float(
            np.mean(edges > 0)
        )


        # 직선 검출
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=35,
            minLineLength=45,
            maxLineGap=5
        )


        horizontal_lines = 0


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
                    angle < 10
                    or
                    angle > 170
                ):

                    horizontal_lines += 1


        # ====================================================
        # 매우 보수적인 안경 판정
        #
        # 맨얼굴 오탐을 줄이기 위해
        # 강한 조건을 동시에 요구
        # ====================================================

        glasses_detected = (
            glasses_count >= 2
            and
            horizontal_lines >= 4
            and
            edge_ratio >= 0.045
        )


    result["glasses"] = glasses_detected


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
    # 최종 부적합
    # ========================================================

    # 안경이 명확하면 즉시 FAIL
    if result["glasses"]:

        result["status"] = "FAIL"

        result["message"] = (
            "안경 또는 선글라스가 감지되었습니다. "
            "안경을 벗어주세요."
        )

        return result


    # 현재 마스크/모자 휴리스틱은
    # 확정 FAIL에 사용하지 않음.
    # 실제 모델을 연결할 때 강화 가능.


    # ========================================================
    # 모든 촬영 조건
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

        self.frame_count = 0


    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )


        result = analyze_frame(
            image
        )


        with self.lock:

            self.latest_result = result

            self.frame_count += 1

            self.pass_history.append(
                result["status"] == "PASS"
            )

            self.fail_history.append(
                result["status"] == "FAIL"
            )


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


        # 얼굴 위치 다시 찾기
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(120, 120)
        )


        if len(faces) > 0:

            x, y, fw, fh = max(
                faces,
                key=lambda box: box[2] * box[3]
            )

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


    st.success(
        "촬영 조건 검사가 완료되었습니다."
    )


    st.markdown(
        """
        ### 🔎 선별검사 단계

        얼굴 촬영 조건이 정상적으로 완료되었습니다.

        이제 실제 선별검사 AI 엔진을 연결할 수 있습니다.
        """,
        unsafe_allow_html=True
    )


    progress = st.progress(0)

    status_box = st.empty()


    # 실제 AI 모델 연결 전 테스트용 진행
    for i in range(101):

        progress.progress(i)

        if i < 30:

            status_box.info(
                "① 영상 품질 확인 중..."
            )

        elif i < 60:

            status_box.info(
                "② 얼굴 특징 데이터 준비 중..."
            )

        elif i < 85:

            status_box.info(
                "③ 선별검사 분석 준비 중..."
            )

        else:

            status_box.success(
                "④ 선별검사 분석 준비 완료"
            )


    st.markdown("---")


    st.markdown(
        """
        <div class="pass">
        ✅ 선별검사 준비 완료
        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        "현재 화면은 선별검사 AI 엔진을 연결하기 위한 단계입니다. "
        "실제 의료 선별 결과를 생성하는 모델은 별도로 연결해야 합니다."
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
