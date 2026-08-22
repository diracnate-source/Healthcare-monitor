import streamlit as st
import cv2
import numpy as np

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

from av import VideoFrame


# ============================================================
# 1. Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="안경 착용 엄격 검사",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 AI 비접촉 멀티모달 뇌 건강 모니터링 시스템")

st.markdown("---")

st.info(
    "촬영 전 안경·선글라스·마스크·모자를 벗고 "
    "정면을 바라봐 주세요.\n\n"
    "안경 착용이 감지되거나 판정이 불확실하면 "
    "촬영을 허용하지 않습니다."
)


# ============================================================
# 2. OpenCV 환경 검사
# ============================================================

def check_opencv():

    required_functions = [
        "CascadeClassifier",
        "cvtColor",
        "equalizeHist",
        "rectangle",
        "putText",
    ]

    missing = []

    for name in required_functions:
        if not hasattr(cv2, name):
            missing.append(name)

    if missing:
        st.error("❌ OpenCV 초기화에 실패했습니다.")

        st.code(
            f"""
OpenCV version:
{getattr(cv2, "__version__", "unknown")}

OpenCV path:
{getattr(cv2, "__file__", "unknown")}

누락된 기능:
{", ".join(missing)}
"""
        )

        st.warning(
            "현재 Python 환경에서 OpenCV가 정상적으로 로딩되지 않았습니다. "
            "requirements.txt의 OpenCV 패키지를 확인하세요."
        )

        return False

    return True


if not check_opencv():
    st.stop()


# ============================================================
# 3. Haar Cascade 경로
# ============================================================

try:

    CASCADE_DIR = cv2.data.haarcascades

except Exception as e:

    st.error("❌ OpenCV Haar Cascade 경로를 찾을 수 없습니다.")

    st.code(str(e))

    st.stop()


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


# ============================================================
# 4. Cascade 로딩
# ============================================================

try:

    face_cascade = cv2.CascadeClassifier(
        FACE_XML
    )

    eye_cascade = cv2.CascadeClassifier(
        EYE_XML
    )

    glasses_cascade = cv2.CascadeClassifier(
        GLASSES_XML
    )

except Exception as e:

    st.error(
        "❌ OpenCV CascadeClassifier 초기화 오류"
    )

    st.code(
        f"""
오류:
{str(e)}

OpenCV:
{getattr(cv2, "__version__", "unknown")}

Cascade 경로:
{CASCADE_DIR}
"""
    )

    st.stop()


# ============================================================
# 5. Cascade 파일 확인
# ============================================================

if face_cascade.empty():

    st.error(
        "❌ 얼굴 검출 모델을 불러오지 못했습니다."
    )

    st.code(FACE_XML)

    st.stop()


if eye_cascade.empty():

    st.error(
        "❌ 눈 검출 모델을 불러오지 못했습니다."
    )

    st.code(EYE_XML)

    st.stop()


if glasses_cascade.empty():

    st.warning(
        "⚠️ 안경 전용 Cascade를 사용할 수 없습니다."
    )

    st.info(
        "안경 검출은 보수적으로 UNKNOWN 처리됩니다. "
        "불확실한 경우 촬영을 허용하지 않습니다."
    )


# ============================================================
# 6. 얼굴 / 눈 / 안경 분석
# ============================================================

def analyze_frame(frame):

    if frame is None:

        return {
            "decision": "UNKNOWN",
            "reason": "영상 프레임 없음",
        }


    # --------------------------------------------------------
    # BGR → Gray
    # --------------------------------------------------------

    try:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(
            gray
        )

    except Exception as e:

        return {
            "decision": "UNKNOWN",
            "reason": f"영상 처리 오류: {str(e)}",
        }


    # --------------------------------------------------------
    # 얼굴 검출
    # --------------------------------------------------------

    try:

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=5,
            minSize=(140, 140),
        )

    except Exception:

        return {
            "decision": "UNKNOWN",
            "reason": "얼굴 검출 실패",
        }


    if len(faces) == 0:

        return {
            "decision": "UNKNOWN",
            "reason": "얼굴을 찾지 못함",
        }


    # 가장 큰 얼굴 선택
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )


    # --------------------------------------------------------
    # 얼굴 크기 확인
    # --------------------------------------------------------

    if w < 180 or h < 180:

        return {
            "decision": "UNKNOWN",
            "reason": "얼굴이 너무 작음",
        }


    # --------------------------------------------------------
    # 눈 / 안경 검사 영역
    # --------------------------------------------------------

    rx1 = max(
        0,
        int(x + 0.05 * w)
    )

    rx2 = min(
        gray.shape[1],
        int(x + 0.95 * w)
    )

    ry1 = max(
        0,
        int(y + 0.15 * h)
    )

    ry2 = min(
        gray.shape[0],
        int(y + 0.62 * h)
    )


    eye_region = gray[
        ry1:ry2,
        rx1:rx2
    ]


    if eye_region.size == 0:

        return {
            "decision": "UNKNOWN",
            "reason": "눈 영역 생성 실패",
        }


    # ========================================================
    # 안경 Cascade
    # ========================================================

    glasses_count = 0

    try:

        if not glasses_cascade.empty():

            glasses = glasses_cascade.detectMultiScale(
                eye_region,
                scaleFactor=1.04,
                minNeighbors=2,
                minSize=(24, 18),
            )

            glasses_count = len(glasses)

    except Exception:

        glasses_count = 0


    # ========================================================
    # 눈 Cascade
    # ========================================================

    eyes_count = 0

    try:

        eyes = eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.06,
            minNeighbors=4,
            minSize=(22, 18),
        )

        eyes_count = len(eyes)

    except Exception:

        eyes_count = 0


    # ========================================================
    # 안경테 형태 보조 분석
    # ========================================================

    try:

        blurred = cv2.GaussianBlur(
            eye_region,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            45,
            130
        )

        edge_ratio = float(
            np.mean(edges > 0)
        )

    except Exception:

        edge_ratio = 0.0


    # ========================================================
    # 1차 판정
    # ========================================================

    # 안경 Cascade가 잡히면 즉시 부적합 후보
    if glasses_count >= 1:

        return {
            "decision": "GLASSES",
            "reason": (
                f"안경 패턴 검출 "
                f"(cascade={glasses_count})"
            ),
            "glasses": glasses_count,
            "eyes": eyes_count,
            "edge": edge_ratio,
        }


    # --------------------------------------------------------
    # 안경테가 의심되는 경우
    # --------------------------------------------------------

    if edge_ratio >= 0.095:

        return {
            "decision": "UNKNOWN",
            "reason": (
                f"안경테 의심 "
                f"(edge={edge_ratio:.3f})"
            ),
            "glasses": glasses_count,
            "eyes": eyes_count,
            "edge": edge_ratio,
        }


    # --------------------------------------------------------
    # 양쪽 눈이 확인되면 무안경 후보
    # --------------------------------------------------------

    if eyes_count >= 2:

        return {
            "decision": "NO_GLASSES",
            "reason": (
                f"양쪽 눈 확인 "
                f"(eyes={eyes_count})"
            ),
            "glasses": glasses_count,
            "eyes": eyes_count,
            "edge": edge_ratio,
        }


    # --------------------------------------------------------
    # 그 외
    # --------------------------------------------------------

    return {
        "decision": "UNKNOWN",
        "reason": (
            f"눈 검출 불충분 "
            f"(eyes={eyes_count})"
        ),
        "glasses": glasses_count,
        "eyes": eyes_count,
        "edge": edge_ratio,
    }


# ============================================================
# 7. WebRTC 영상 처리 클래스
# ============================================================

class GlassesProcessor(VideoProcessorBase):

    def __init__(self):

        self.frames = 0

        self.glasses_votes = 0

        self.no_glasses_votes = 0

        self.unknown_votes = 0

        self.last_decision = "UNKNOWN"

        self.last_reason = "카메라 준비 중"


    def recv(
        self,
        frame: VideoFrame
    ) -> VideoFrame:

        try:

            img = frame.to_ndarray(
                format="bgr24"
            )

        except Exception:

            return frame


        # ----------------------------------------------------
        # 분석
        # ----------------------------------------------------

        result = analyze_frame(
            img
        )


        self.frames += 1

        self.last_decision = (
            result["decision"]
        )

        self.last_reason = (
            result["reason"]
        )


        # ----------------------------------------------------
        # 투표 누적
        # ----------------------------------------------------

        if result["decision"] == "GLASSES":

            self.glasses_votes += 1

        elif result["decision"] == "NO_GLASSES":

            self.no_glasses_votes += 1

        else:

            self.unknown_votes += 1


        # ====================================================
        # 화면 표시
        # ====================================================

        h, w = img.shape[:2]


        x1 = int(w * 0.20)
        y1 = int(h * 0.10)

        x2 = int(w * 0.80)
        y2 = int(h * 0.90)


        # ----------------------------------------------------
        # 색상 및 문구
        # ----------------------------------------------------

        if self.last_decision == "GLASSES":

            color = (
                0,
                0,
                255
            )

            text = (
                "NOT ELIGIBLE - GLASSES"
            )


        elif self.last_decision == "NO_GLASSES":

            color = (
                0,
                220,
                0
            )

            text = (
                "NO GLASSES - CHECKING"
            )


        else:

            color = (
                0,
                180,
                255
            )

            text = (
                "CHECKING - DO NOT SHOOT"
            )


        # ----------------------------------------------------
        # 얼굴 가이드 박스
        # ----------------------------------------------------

        try:

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )


            cv2.putText(
                img,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                color,
                2,
                cv2.LINE_AA,
            )

        except Exception:

            pass


        return VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# 8. 여러 프레임 최종 판정
# ============================================================

def get_final_decision(
    processor
):

    if processor is None:

        return "UNKNOWN"


    total = (
        processor.glasses_votes
        + processor.no_glasses_votes
        + processor.unknown_votes
    )


    # 최소 8프레임 필요
    if total < 8:

        return "UNKNOWN"


    # --------------------------------------------------------
    # 안경이 한 번이라도 확실히 검출되면 차단
    # --------------------------------------------------------

    if processor.glasses_votes >= 1:

        return "GLASSES"


    # --------------------------------------------------------
    # 불확실성이 20% 이상이면 차단
    # --------------------------------------------------------

    if (
        processor.unknown_votes
        >
        max(
            1,
            int(total * 0.20)
        )
    ):

        return "UNKNOWN"


    # --------------------------------------------------------
    # 최소 8개 이상의 무안경 프레임
    # --------------------------------------------------------

    if processor.no_glasses_votes >= 8:

        return "NO_GLASSES"


    return "UNKNOWN"


# ============================================================
# 9. Session State
# ============================================================

if "step" not in st.session_state:

    st.session_state.step = "check"


if "capture_allowed" not in st.session_state:

    st.session_state.capture_allowed = False


# ============================================================
# 10. 1단계 — 안경 검사
# ============================================================

if st.session_state.step == "check":

    st.subheader(
        "🔍 1단계: 안경 착용 엄격 검사"
    )


    st.write(
        "카메라를 켜고 얼굴을 가이드 박스 안에 "
        "위치시키세요."
    )


    st.write(
        "안경 착용이 감지되면 즉시 촬영이 차단됩니다."
    )


    st.write(
        "판정이 애매한 경우에도 촬영을 허용하지 않습니다."
    )


    # ========================================================
    # WebRTC
    # ========================================================

    ctx = webrtc_streamer(

        key="strict-glasses-check",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=GlassesProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

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


    # ========================================================
    # 판정 표시
    # ========================================================

    if ctx.video_processor:

        processor = (
            ctx.video_processor
        )


        # ----------------------------------------------------
        # 현재 통계
        # ----------------------------------------------------

        st.caption(
            f"검사 프레임: {processor.frames} | "
            f"안경 검출: {processor.glasses_votes} | "
            f"무안경: {processor.no_glasses_votes} | "
            f"불확실: {processor.unknown_votes}"
        )


        # ----------------------------------------------------
        # 최종 판정
        # ----------------------------------------------------

        final = get_final_decision(
            processor
        )


        # ====================================================
        # 안경 착용
        # ====================================================

        if final == "GLASSES":

            st.error(
                "❌ 부적합 — 안경 착용"
            )


            st.write(
                "안경을 벗은 후 다시 검사해주세요."
            )


            st.session_state.capture_allowed = False


        # ====================================================
        # 무안경
        # ====================================================

        elif final == "NO_GLASSES":

            st.success(
                "✅ 촬영 가능합니다"
            )


            st.write(
                "안경 미착용 상태가 확인되었습니다."
            )


            st.session_state.capture_allowed = True


        # ====================================================
        # 불확실
        # ====================================================

        else:

            st.warning(
                "⚠️ 검사 중 — 아직 촬영할 수 없습니다."
            )


            st.write(
                "정면을 바라보고 얼굴을 가이드 박스 "
                "안에 위치시켜 주세요."
            )


            st.session_state.capture_allowed = False


    # ========================================================
    # 촬영 버튼
    # ========================================================

    st.markdown("---")


    if st.session_state.capture_allowed:

        if st.button(
            "📸 촬영하기",
            type="primary",
            use_container_width=True,
        ):

            st.session_state.step = "result"

            st.rerun()


    else:

        st.button(
            "🔒 촬영 불가 — 안경 검사 통과 필요",
            disabled=True,
            use_container_width=True,
        )


    st.caption(
        "※ 현재 안경 검사는 OpenCV 기반 사전검사입니다. "
        "100% 정확한 안경 분류를 보장하는 의료용 AI 모델은 아닙니다."
    )


# ============================================================
# 11. 2단계 — 촬영 후 분석 화면
# ============================================================

elif st.session_state.step == "result":

    st.subheader(
        "📋 2단계: 비접촉 멀티모달 뇌 건강 모니터링"
    )


    st.success(
        "✅ 촬영 조건을 통과했습니다."
    )


    st.info(
        "안경 착용 검사 통과 후 다음 단계의 "
        "비접촉 측정을 진행할 수 있습니다."
    )


    # ========================================================
    # 지표
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "언어 유창성 지표",
            "측정 대기"
        )


        st.metric(
            "머리 안정도 지표",
            "측정 대기"
        )


    with col2:

        st.metric(
            "발화 리듬 지표",
            "측정 대기"
        )


        st.metric(
            "인지 반응성 지표",
            "측정 대기"
        )


    st.markdown("---")


    st.warning(
        "⚠️ 현재 지표는 시연용입니다. "
        "실제 뇌 건강 또는 치매 위험도를 판단하려면 "
        "검증된 임상 알고리즘과 데이터가 필요합니다."
    )


    # ========================================================
    # 처음으로
    # ========================================================

    if st.button(
        "🔄 안경 검사 다시 하기",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.step = "check"

        st.session_state.capture_allowed = False

        st.rerun()
