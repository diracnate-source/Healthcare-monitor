import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from av import VideoFrame

st.set_page_config(page_title='AI 비접촉 멀티모달 뇌 건강 모니터링', page_icon='🧠', layout='centered')
st.title('🧠 AI 비접촉 멀티모달 뇌 건강 모니터링 시스템')
st.markdown('---')
st.info('촬영 전 안경·선글라스·마스크·모자를 벗고 정면을 바라봐 주세요. 안경 착용이 의심되거나 판정이 불확실하면 촬영을 허용하지 않습니다.')

FACE_XML=cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
EYE_XML=cv2.data.haarcascades+'haarcascade_eye.xml'
GLASSES_XML=cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml'
face_cascade=cv2.CascadeClassifier(FACE_XML)
eye_cascade=cv2.CascadeClassifier(EYE_XML)
glasses_cascade=cv2.CascadeClassifier(GLASSES_XML)
if face_cascade.empty() or eye_cascade.empty() or glasses_cascade.empty():
    st.error('OpenCV Haar 모델을 불러오지 못했습니다.')
    st.stop()

def analyze_frame(frame):
    if frame is None: return {'decision':'UNKNOWN','reason':'영상 프레임 없음'}
    gray=cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    faces=face_cascade.detectMultiScale(gray,1.10,5,minSize=(140,140))
    if len(faces)==0: return {'decision':'UNKNOWN','reason':'얼굴을 찾지 못함'}
    x,y,w,h=max(faces,key=lambda f:f[2]*f[3])
    if w<180 or h<180: return {'decision':'UNKNOWN','reason':'얼굴이 너무 작음'}
    rx1,rx2=max(0,int(x+.06*w)),min(gray.shape[1],int(x+.94*w))
    ry1,ry2=max(0,int(y+.16*h)),min(gray.shape[0],int(y+.60*h))
    eye_region=gray[ry1:ry2,rx1:rx2]
    if eye_region.size==0: return {'decision':'UNKNOWN','reason':'눈 영역 생성 실패'}
    glasses=glasses_cascade.detectMultiScale(eye_region,1.04,2,minSize=(24,18))
    eyes=eye_cascade.detectMultiScale(eye_region,1.06,4,minSize=(22,18))
    edges=cv2.Canny(cv2.GaussianBlur(eye_region,(5,5),0),45,130)
    edge_ratio=float(np.mean(edges>0))
    if len(glasses)>=1: return {'decision':'GLASSES','reason':f'안경 패턴 검출(cascade={len(glasses)})'}
    if edge_ratio>=.095: return {'decision':'UNKNOWN','reason':f'안경테 의심(edge={edge_ratio:.3f})'}
    if len(eyes)>=2: return {'decision':'NO_GLASSES','reason':f'양쪽 눈 확인(eyes={len(eyes)})'}
    return {'decision':'UNKNOWN','reason':f'눈 검출 불충분(eyes={len(eyes)})'}

class GlassesProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames=0; self.glasses_votes=0; self.no_glasses_votes=0; self.unknown_votes=0
        self.last_decision='UNKNOWN'; self.last_reason='카메라 준비 중'
    def recv(self,frame):
        img=frame.to_ndarray(format='bgr24'); r=analyze_frame(img)
        self.frames+=1; self.last_decision=r['decision']; self.last_reason=r['reason']
        if r['decision']=='GLASSES': self.glasses_votes+=1
        elif r['decision']=='NO_GLASSES': self.no_glasses_votes+=1
        else: self.unknown_votes+=1
        h,w=img.shape[:2]; x1,y1,x2,y2=int(w*.20),int(h*.10),int(w*.80),int(h*.90)
        if self.last_decision=='GLASSES': color,text=(0,0,255),'NOT ELIGIBLE - GLASSES'
        elif self.last_decision=='NO_GLASSES': color,text=(0,200,0),'NO GLASSES - CHECKING'
        else: color,text=(0,180,255),'CHECKING - DO NOT SHOOT'
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2); cv2.putText(img,text,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.72,color,2,cv2.LINE_AA)
        return VideoFrame.from_ndarray(img,format='bgr24')

def get_final_decision(p):
    total=p.glasses_votes+p.no_glasses_votes+p.unknown_votes
    if total<8: return 'UNKNOWN'
    if p.glasses_votes>=1: return 'GLASSES'
    if p.unknown_votes>max(1,int(total*.20)): return 'UNKNOWN'
    if p.no_glasses_votes>=8: return 'NO_GLASSES'
    return 'UNKNOWN'

if 'step' not in st.session_state: st.session_state.step='check'
if 'capture_allowed' not in st.session_state: st.session_state.capture_allowed=False

if st.session_state.step=='check':
    st.subheader('🔍 1단계: 촬영 조건 사전 검사')
    st.write('카메라를 켜고 얼굴을 가이드 박스 안에 위치시키세요. 불확실하면 촬영을 허용하지 않습니다.')
    ctx=webrtc_streamer(key='strict-glasses-check',mode=WebRtcMode.SENDRECV,video_processor_factory=GlassesProcessor,media_stream_constraints={'video':True,'audio':False},rtc_configuration={'iceServers':[{'urls':['stun:stun.l.google.com:19302']}]})
    if ctx.video_processor:
        p=ctx.video_processor
        st.caption(f'프레임: {p.frames} | 안경: {p.glasses_votes} | 무안경: {p.no_glasses_votes} | 불확실: {p.unknown_votes}')
        final=get_final_decision(p)
        if final=='GLASSES':
            st.error('❌ 부적합 — 안경 착용이 감지되었습니다. 안경을 벗은 후 다시 검사하세요.'); st.session_state.capture_allowed=False
        elif final=='NO_GLASSES':
            st.success('✅ 촬영 가능합니다 — 안경 미착용 상태가 확인되었습니다.'); st.session_state.capture_allowed=True
        else:
            st.warning('⚠️ 아직 판정이 충분히 확실하지 않습니다. 정면을 바라보고 잠시 기다려 주세요.'); st.session_state.capture_allowed=False
    st.markdown('---')
    if st.session_state.capture_allowed:
        if st.button('📸 촬영 조건 통과 → 분석 시작',type='primary',use_container_width=True): st.session_state.step='result'; st.rerun()
    else: st.button('🔒 촬영 차단됨 — 안경을 벗고 다시 검사',disabled=True,use_container_width=True)
    st.caption('배포 안정성을 위해 PyTorch/CLIP을 제거한 경량 사전검사 버전입니다. 안경 검출의 임상적 100% 정확도를 보장하는 모델은 아닙니다.')
else:
    st.subheader('📋 2단계: 비접촉 멀티모달 뇌 건강 모니터링')
    st.success('✅ 촬영 조건 사전 검사를 통과했습니다.')
    st.info('현재 버전은 배포 안정성을 위해 사전검사와 결과 화면까지 구현했습니다. 실제 뇌 건강 위험도는 검증된 임상 알고리즘과 실제 측정 데이터를 연결해야 합니다.')
    c1,c2=st.columns(2)
    with c1:
        st.metric('언어 유창성 지표 (S)','측정 대기'); st.metric('머리 안정도 지표 (P)','측정 대기')
    with c2:
        st.metric('발화 리듬 지표 (B)','측정 대기'); st.metric('인지 반응성 지표 (Θ)','측정 대기')
    st.warning('⚠️ 현재 표시되는 지표는 임상 결과가 아닙니다. 실제 의료적 판단에는 검증된 데이터와 임상 알고리즘이 필요합니다.')
    if st.button('🔄 처음으로 돌아가기',type='primary',use_container_width=True): st.session_state.step='check'; st.session_state.capture_allowed=False; st.rerun()
