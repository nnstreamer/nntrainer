# B/C 관측 데이터 v1

각 train200 / validation80 / test80. A 제외. 입력은 time/moving/calendar/notification/recent_apps 다섯 개다.
최근 앱 네 개는 과거→최근 순서, 알림은 화면 원문 한 개이며 null은 없음이다. 날씨·알람·정산/관계/요청 분류는 없다.
B: 개인사업자, 네이버지도·사업 이체 기업은행·개인 송금 카카오페이.
C: 대학생, 카카오맵/YouTube 교대·모든 송금 토스. 사람별 LoRA 하나로 전체 상황을 학습한다.

`by_user/<user>/<split>.jsonl`: 감독/평가용, `native_questions`에는 정답 없는 decision 입력, `labels`에는 별도 정답.
`input.schema.json`과 `app_catalog.json`은 고정 계약이다. J는 시계 대신 기업은행이며 이전 adapter를 혼용할 수 없다.
학습 이동48행은12관측×4감독이며 C 두 앱3:1이다. Validation/test80행은 각각 고유80관측이고 반복 행은 없다.
알림 원문이 단서인 송금과 답장, 무관하거나 없는 알림 속의 일정/이동/이력 판단을 함께 포함한다.
알림만 보고 정답을 유일하게 정할 수 없는 대조가 있다. 정확한 분포/토큰 길이/파일 hash는 manifest를 본다.

이것은 명확한 합성 데모이며 실제 사용자 이력에서 일반화를 입증하지 않는다. 모델 확률은 개인화를 위한 관측 정책의 학습 결과다.
스키마 설명: ../../docs/schemas/mobile_context_bc_v1.md
사전 실험: ../../docs/experiments/20261016_bc_observed.md
