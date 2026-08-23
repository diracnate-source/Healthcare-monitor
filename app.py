import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(
    page_title="카메라 테스트",
    page_icon="📷"
)

st.title("📷 카메라 연결 테스트")

st.write("START를 누르고 브라우저에서 카메라 권한을 허용하세요.")

def video_frame_callback(frame):
    return frame

ctx = webrtc_streamer(
    key="camera_test",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
)

if ctx.state.playing:
    st.success("🟢 카메라 연결 성공")
else:
    st.info("📷 START 버튼을 눌러주세요.")
