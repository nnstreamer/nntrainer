# B/C 관측 데이터와 Python 참조 실험

이 문서는 전송용 데모에 필요한 원본 데이터/선택/학습 계약을 요약한다. 원본 데이터 manifest는 `data/personalization_bc_observed_v1/manifest.json`이며 데이터의 내용은 최초 생성본과 같다. 파일 hash는 `python tools/verify_bundle.py`로 검증한다.

- 모델: anthonym21/qwen3-0.6b-rlcd-decision, revision b327ec5efb5fdbf8bfafa3b369720ac5f6434b05.
- B 개인사업자: 네이버지도, 사업 이체 기업은행, 개인 송금 카카오페이. C 대학생: 카카오맵/YouTube 교대, 모든 송금 토스.
- 입력: 시간/이동 여부/당일 일정/최근 화면 알림 원문 한 건/최근 앱 네 개. 관계/정산/요청 상태 분류·날씨·알람은 없다. 스키마의 허용 키를 확장하지 않는다.
- 원본 데이터: 각 train200/validation80/test80. 학습 고유 관측164개, 이동48행은12관측×4감독으로 C 두 앱3:1을 표현한다. Test는 반복 없는80관측이다. A는 없다.
- 전체 프롬프트153–180tokens. 실제 기기 숫자 입력은 원본 tokenizer/decision prompt로 사전 변환했고 Transformers 없이 실행할 수 있다. `input_ids`와 one-hot `target_probs`/`primary_label_index`를 포함한다.
- 시연 네 개는 원본 validation에서 과거 세 앱 이력 사용 횟수와 event_id로 선택했다. 원시 모델 출력이나 test 성공률로 고르지 않았다. `data/scenario-selection.json`에 선택 근거가 있다.
- Python 참조는 QS4CX base의 FP32 수치 계산과 per-output-channel INT4 LoRA QAT, rank8/alpha16·마지막4block28FC·고정 head·batch2/lr2e-4/seed42다. B3epoch/300updates는 사업 Top1 8/8·개인0/8·전체 Hit@4 85%였다. B7epoch/700updates는 사업6/8·개인8/8·전체97.5%였다. 같은 seed의 새7epoch 재학습에서도 확률 차이0으로 재현됐다. C4epoch는 전체86.25%, 카카오맵/YouTube 동시10/12다.
- 현재 시연의 B 두 validation 입력은 Python7epoch에서 기업은행99.82%/카카오페이99.83%로 Top1 정답이었다. 이것은 새 기기의 실행 결과가 아니다.

기기는1epoch/batch4를 기본으로 제공하며 사용자 설정으로 변경한다. Python의 같은 epoch 수와는 update 수가 다르다. 새 기기의 정확도/속도는 실제 실행으로 확인해야 한다. 이 패키지에 학습 adapter·모델 가중치·과거 기기의 실시간 출력은 포함하지 않는다.

합성 PoC의 제한: 과거 세 앱은 학습12조합/평가4조합뿐이며 설명용 samples.md는 j=0을 골라 모두 같은 과거 세 앱이다. 문구/시간 분포가 좁고 이미 공개한 test·한 seed를 재사용했다. C 일정 Hit@4 65.6%와 NB93.75% 대비 모델86.25%의 열세를 숨기지 않는다. 실제 사용자 일반화나 모든 단순 추천보다 우월하다는 증거는 아니다.
