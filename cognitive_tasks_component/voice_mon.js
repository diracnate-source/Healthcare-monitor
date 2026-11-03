/* 공통 음성 모니터: 마이크 소리 '크기'만 20ms 간격으로 계산한다. 녹음·저장·전사는 하지 않는다.
   말 시작(onset) = 소음 바닥+10dB(절대 -55dBFS 이상)을 80ms 이상 이어서 넘긴 첫 시점.
   말 끝 = 200ms 이상 쉰 시점. 120ms 미만 구간은 잡음으로 버린다. 연구용 시제품. */
(function(){
  const FRAME=20, ONSET_RUN=4, END_GAP_MS=200, MIN_SEG=120, MARGIN=10, ABS_MIN=-55;
  const pct=(a,p)=>{const s=[...a].sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.max(0,Math.floor(p*(s.length-1))))];};
  const VM={status:"idle",error:null,subs:new Set(),segs:[],lv:[],lvT0:null,buf:[],floor:-60,thr:-45,
    run:0,runStart:null,inSeg:false,segStart:null,gap:0,lastLoud:0,tick:0,onLevel:null,openStart:null};
  window.__fakeDb=null;      // 시험용: 숫자를 넣으면 실제 마이크 대신 이 dB를 쓴다
  VM._sample=function(t,d){
    if(window.__fakeDb!==null&&window.__fakeDb!==undefined) d=window.__fakeDb;
    if(VM.lvT0===null) VM.lvT0=t;
    VM.lv.push(Math.max(-100,Math.round(d)));
    VM.buf.push(d); if(VM.buf.length>400) VM.buf.shift();
    if((VM.tick++%10)===0 && VM.buf.length>=50){VM.floor=pct(VM.buf,0.2);VM.thr=Math.max(VM.floor+MARGIN,ABS_MIN);}
    const loud=d>VM.thr;
    if(VM.onLevel) VM.onLevel(d);
    if(loud){
      VM.run++; if(VM.run===1) VM.runStart=t;
      VM.lastLoud=t; VM.gap=0;
      if(VM.run>=ONSET_RUN && !VM.inSeg){VM.inSeg=true;VM.segStart=VM.runStart;VM.openStart=VM.segStart;
        for(const f of [...VM.subs]) f(VM.segStart);}
    }else{
      if(VM.inSeg){VM.gap+=FRAME; if(VM.gap>=END_GAP_MS){const e=VM.lastLoud+FRAME; if(e-VM.segStart>=MIN_SEG) VM.segs.push({s:VM.segStart,e:e}); VM.inSeg=false;VM.openStart=null;VM.run=0;}}
      else VM.run=0;
    }
  };
  VM.start=async function(){
    if(VM.status==="ok"||VM.status==="starting") return VM.status;
    VM.status="starting";
    try{
      if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia) throw {name:"unavailable"};
      const stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
      const AC=window.AudioContext||window.webkitAudioContext, ctx=new AC();
      if(ctx.state==="suspended"){try{await ctx.resume();}catch(_){}}
      const an=ctx.createAnalyser(); an.fftSize=1024; ctx.createMediaStreamSource(stream).connect(an);
      const buf=new Float32Array(an.fftSize); let next=performance.now();
      VM._stream=stream; VM._ctx=ctx;
      setInterval(()=>{const n=performance.now(); an.getFloatTimeDomainData(buf);
        let s=0; for(let i=0;i<buf.length;i++) s+=buf[i]*buf[i];
        const d=20*Math.log10(Math.sqrt(s/buf.length)+1e-8);
        while(next<=n){VM._sample(next,d);next+=FRAME;}
      },10);
      VM.status="ok";
    }catch(e){
      VM.status=(e&&(e.name==="NotAllowedError"||e.name==="SecurityError"))?"denied":"unavailable"; VM.error=String(e&&e.name||e);
    }
    return VM.status;
  };
  VM.ok=()=>VM.status==="ok";
  VM.subscribe=f=>{VM.subs.add(f);return ()=>VM.subs.delete(f);};
  // [a,b] (performance.now 기준 ms)와 겹치는 말 구간 (구간 밖은 잘라 낸다)
  VM.segments=function(a,b){
    const all=VM.segs.slice(); if(VM.inSeg&&VM.openStart!==null) all.push({s:VM.openStart,e:performance.now()});
    return all.filter(x=>x.e>a&&x.s<b).map(x=>({s:Math.max(x.s,a),e:Math.min(x.e,b)}));
  };
  VM.features=function(a,b){
    const s=VM.segments(a,b);
    return {n_segments:s.length,first_onset_ms:s.length?Math.round((s[0].s-a)*10)/10:null,
            speech_time_s:Math.round(s.reduce((x,y)=>x+y.e-y.s,0)/100)/10};
  };
  VM.levels=function(a,b){            // a~b 사이 20ms 격자 소리 크기(dBFS)
    if(VM.lvT0===null) return [];
    const i=Math.max(0,Math.floor((a-VM.lvT0)/FRAME)), j=Math.min(VM.lv.length,Math.floor((b-VM.lvT0)/FRAME));
    return VM.lv.slice(i,j);
  };
  VM.allSegmentEvents=function(){return VM.segs.map(x=>({task:"voice",trial:0,event:"seg",
      t_epoch_ms:Math.round(performance.timeOrigin+x.s),end_epoch_ms:Math.round(performance.timeOrigin+x.e)}));};
  VM.toPerf=ep=>ep-performance.timeOrigin;
  window.VM=VM;
})();
