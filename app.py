import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(
    page_title="카메라 테스트",
    page_icon="📷"
)

st.title("📷 웹캠 연결 테스트")

st.write("START 버튼을 누르고 브라우저에서 카메라 사용을 허용하세요.")

ctx = webrtc_streamer(
    key="camera-test-01",
    mode=WebRtcMode.SENDRECV,
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
    async_processing=True,
)

if ctx.state.playing:
    st.success("🟢 카메라 연결 성공")

else:
    st.info("📷 START 버튼을 눌러주세요.")
