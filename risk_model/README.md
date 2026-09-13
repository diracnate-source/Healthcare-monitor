# risk_model — 학습 기반 통합 위험도 산출 모듈

`6개 지표 + 인구통계(나이 등) + 개인/모집단 통계 -> 학습된 모델 -> risk_score`
구조를 구현한 모듈입니다. 기존 v2(경험적 베이즈 + 단측 Z-score + 지수 스쿼싱)
알고리즘은 폐기하지 않고 `legacy_scoring.py`에 그대로 남겨, ① 콜드스타트
폴백 ② 새 모델과의 성능 비교 기준선(baseline) 두 용도로 계속 사용합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `legacy_scoring.py` | v2 알고리즘 원본 로직 (Welford, EB 결합, 단측 Z, 지수 스쿼싱) |
| `features.py` | 원시 지표 + 인구통계 + 개인/모집단 통계 → 학습모델 입력 피처 벡터 |
| `model.py` | `RiskModel` 클래스 (학습/추론/저장/로드) + `score_with_fallback` 서비스 진입점 |
| `train.py` | 라벨된 CSV로 학습 → 교차검증 → legacy 대비 성능 비교 → 저장 (CLI) |

## 왜 이렇게 설계했는가

1. **원시값 + 편차값(z)을 함께 피처로 넣는다.** legacy처럼 사람이 미리
   z-score를 계산해 고정 가중합만 하는 대신, 모델이 원시값과 개인/모집단
   대비 편차를 동시에 보고 지표 간 비선형 상호작용을 스스로 학습하게 했습니다.
   (지난 논의에서 지적된 "독립 계산 후 단순 가중합 = 단순 집합" 문제를
   구조적으로 해결하는 부분입니다.)
2. **결측은 임의 대치 대신 플래그로 남긴다.** `{metric}_missing` 피처로
   "이 값이 원래 없었다"는 정보 자체를 모델이 활용할 수 있게 합니다.
3. **`n_personal`을 피처로 직접 넣는다.** 콜드스타트 정도(개인 데이터가
   얼마나 쌓였는지)를 모델이 스스로 반영하게 해, EB의 `K` 상수를 사람이
   고정하지 않아도 되게 합니다.
4. **"학습됨" ≠ "신뢰할 수 있음".** `RiskModel.is_reliable()`은 단순히
   모델이 fit()되었는지가 아니라, (a) 최소 샘플 수 이상으로 학습되었고
   (b) 별도 검증에서 legacy보다 AUC가 유의미하게(현재 임계값 +0.02) 높은지
   까지 확인합니다. 둘 중 하나라도 못 채우면 `score_with_fallback()`이
   자동으로 legacy로 폴백합니다 — "구조를 AI로 바꿨으니 성능이 확실히
   좋아졌다"고 검증 없이 주장하지 않기 위한 안전장치입니다.

## 사용법

```bash
# 1) 학습 (라벨 데이터가 모인 뒤)
python -m risk_model.train \
    --data labeled_sessions.csv \
    --population-stats population_stats.json \
    --personal-stats personal_stats.json \
    --out-dir ./trained_model

# 2) 서비스 코드에서 추론
from risk_model.model import RiskModel, score_with_fallback
from risk_model.features import RawSample, build_feature_frame

model = RiskModel.load("./trained_model")  # 없으면 None으로 두면 자동 legacy 폴백
result = score_with_fallback(model, feature_row, raw_metrics, population_stats, personal_stats)
# result["mode"] 로 어떤 경로(learned_model / fallback_bootstrap / fallback_data_driven)로
# 산출됐는지 항상 확인 가능
```

`labeled_sessions.csv` 필요 컬럼: `sample_id, user_id, expression_change,
micro_movement, facial_asymmetry, blink_rate, gaze_variability, reaction_ms,
age, sex, education_years, n_personal, label`
(`label`은 진단 또는 MMSE 컷오프 기준 이진값 — 정의는 임상 담당자와 협의 필요)

## 배포 전 체크리스트 (지난 논의 반영)

- [ ] 라벨 데이터가 `MIN_TRAINING_SAMPLES`(기본 150건, 도메인에 맞게 조정) 이상인가
- [ ] 별도 홀드아웃/외부 코호트로 legacy 대비 AUC·민감도·특이도 개선을 확인했는가
   (같은 데이터로 교차검증만 한 결과는 과최적화 위험이 있으므로 최종 판단에는
   부족합니다)
- [ ] 확률 보정(calibration curve)이 실제 위험 비율과 맞는지 확인했는가
- [ ] `is_validated_better_than_legacy`가 실제 코드 상에서 `False`인 채로
      배포되지 않는지 (기본값은 안전하게 `False`이며, `train.py`가 검증
      결과를 보고 자동으로 세팅합니다 — 수동으로 덮어쓰지 마세요)
- [ ] "AI 기반"이라는 표현을 문서/마케팅에 쓸 경우, 실제로 학습된 모델이
      `is_reliable() == True`로 서비스 트래픽에 반영되고 있는 상태인지 확인
      (콜드스타트로 legacy 폴백 중인데 "AI 알고리즘"이라고 표기하면 사실과
      어긋납니다)
- [ ] SaMD(디지털 치료/진단 보조기기) 규제 해당 여부를 규제 전문가와 확인

## 알려진 제약 (다음 반복에서 개선 후보)

- 현재 베이스라인은 로지스틱 회귀입니다. 라벨이 충분히 쌓이면(수백~수천 건)
  `train.py`의 `_build_pipeline()`을 Gradient Boosting(XGBoost/LightGBM) 또는
  사용자를 랜덤효과로 넣는 혼합효과모델로 교체하는 것을 권장합니다.
- 다기관/다인구집단 데이터를 합칠 경우, 기관 간 카메라·조명 조건 차이로
  인한 분포 이동(domain shift)을 별도로 점검해야 합니다.
- 본 모듈은 진단 도구가 아닌 프로토타입이며, 여기서 산출되는 risk_score는
  임상적으로 검증되지 않았습니다.
