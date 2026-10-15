"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const data = window.DEMO_DATA;
  const apps = Object.fromEntries(
    data.candidates.map((app) => [app.app_id, app])
  );
  const profiles = {
    u01: ["사용자 A", "네이버 지도 · 토스 · 데이터 준비 / 미학습"],
    u02: ["사용자 B", "카카오맵 · 카카오페이 · 가족 정산은 나중에"],
    u03: ["사용자 C", "이동 중 카카오맵 + YouTube를 함께 추천"],
  };
  const cases = {
    map_trip: [
      "약속 장소로 이동",
      {
        departing: "출발 전",
        enroute: "이동 중",
        arrived: "도착 후",
        plan: "나중 약속",
      },
    ],
    payment_request: [
      "송금 요청",
      {
        now: "즉시 송금",
        later: "나중에 송금",
        done: "정산 완료",
        account: "계좌 회신",
      },
    ],
    contact_request: [
      "상대별 정산 · 복귀",
      {
        family_now: "가족 · 즉시",
        work_now: "동료 · 즉시",
        family_later: "가족 · 나중에",
        work_later: "동료 · 나중에",
        family_done: "가족 · 완료",
        work_done: "동료 · 완료",
        family_account: "가족 · 계좌",
        work_account: "동료 · 계좌",
      },
    ],
    meeting_lifecycle: [
      "회의 전후",
      {
        upcoming: "시작 전",
        join: "회의 시작",
        ended: "종료 후",
        cancelled: "회의 취소",
      },
    ],
  };
  const translations = {
    home: "집",
    office: "회사",
    transit: "이동 중",
    outdoors: "실외",
    meeting_location: "약속 장소",
    cafe: "카페",
    stationary: "머무는 중",
    preparing_to_leave: "출발 준비",
    travelling: "이동 중",
    arrived: "도착",
    friend: "친구",
    family: "가족",
    work: "동료",
    colleague: "동료",
    coworker: "동료",
    now: "즉시",
    later: "나중에",
    none: "없음",
    pending: "미처리",
    done: "완료",
    completed: "완료",
    account_requested: "계좌 회신 요청",
    account: "계좌 회신",
    scheduled: "예정",
    active: "진행 중",
    starting: "시작",
    ended: "종료",
    enroute: "이동 중",
    departing: "출발 전",
    plan: "나중 약속",
    upcoming: "시작 전",
    join: "회의 시작",
    cancelled: "취소",
    clear: "맑음",
    cloudy: "흐림",
    rain: "비",
    snow: "눈",
    cold: "추움",
    cool: "선선함",
    mild: "온화함",
    warm: "따뜻함",
    hot: "더움",
  };
  const tr = (value) => (value == null ? "없음" : translations[value] || value);
  const state = {
    ...data.default_selection,
    resultMode: "answer",
    live: null,
    generation: 0,
  };
  const params = new URLSearchParams(location.search);
  const udid = params.get("udid") || "R3CY10WM83Y";
  const base = new URL(params.get("server") || "http://localhost:8002/");
  if (!["http:", "https:"].includes(base.protocol))
    throw Error("미러링 서버 주소는 http:// 또는 https://로 지정하세요.");
  base.search = "";
  base.hash = "";
  const wsUrl = new URL(base);
  wsUrl.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  wsUrl.search = new URLSearchParams({
    action: "proxy-adb",
    remote: "tcp:8886",
    udid,
  }).toString();
  base.hash =
    "!" +
    new URLSearchParams({
      action: "stream",
      udid,
      player: "mse",
      fitToScreen: "true",
      ws: wsUrl.toString(),
    });
  $("mirror").src = base.toString();

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = text;
    return el;
  }

  // Locally drawn app marks, accompanied by names; no external icon requests.
  const marks = {
    naver_map: [
      "#03c75a",
      '<path d="M6 6h4l5 8V6h4v14h-4l-5-8v8H6z" fill="white"/>',
    ],
    kakao_map: [
      "#ffe500",
      '<path d="M12 3a7 7 0 0 0-7 7c0 5 7 12 7 12s7-7 7-12a7 7 0 0 0-7-7" fill="#2585ee"/><circle cx="12" cy="10" r="2.6" fill="white"/>',
    ],
    kakao_t: ["#243854", '<path d="M4 5h16v4h-6v12h-4V9H4z" fill="#ffe500"/>'],
    kakaotalk: [
      "#fee500",
      '<path d="M12 4c-5 0-9 3-9 7 0 3 2 5 5 6l-1 4 5-3c5 0 9-3 9-7s-4-7-9-7" fill="#3b2927"/><text x="12" y="13" text-anchor="middle" font-family="sans-serif" font-size="6" fill="#fee500">TALK</text>',
    ],
    toss: [
      "#2668f5",
      '<path d="M5 9c3-7 12-7 15 0 2 5-3 10-8 9-4-1-8-5-7-9" fill="white"/>',
    ],
    kakaopay: [
      "#ffdf00",
      '<path d="M4 7h7v9H4l-1 3v-3H2V7z" fill="#272727"/><text x="16" y="15" text-anchor="middle" font-family="sans-serif" font-size="8" font-weight="bold" fill="#272727">₩</text>',
    ],
    youtube: [
      "#fff",
      '<rect x="2" y="5" width="20" height="14" rx="5" fill="#f03"/><path d="m10 9 6 3-6 3z" fill="white"/>',
    ],
    calendar_app: [
      "#4285f4",
      '<path d="M5 5h14v15H5z" fill="white"/><path d="M5 5h14v4H5z" fill="#76a7ff"/><text x="12" y="17" text-anchor="middle" font-family="sans-serif" font-size="9" font-weight="bold" fill="#4285f4">31</text>',
    ],
    meet: [
      "#fff",
      '<rect x="3" y="6" width="12" height="13" rx="2" fill="#21a467"/><path d="m15 10 6-4v14l-6-4z" fill="#4c8bf5"/>',
    ],
    clock: [
      "#5b65df",
      '<circle cx="12" cy="12" r="8" fill="white"/><path d="M12 6v6l4 2" fill="none" stroke="#515ac5" stroke-width="2" stroke-linecap="round"/>',
    ],
    anki: [
      "#273e65",
      '<path d="m12 2 3 7 8 1-6 5 2 8-7-4-7 4 2-8-6-5 8-1z" fill="#8bd4ff"/>',
    ],
    notes: [
      "#ee8557",
      '<rect x="5" y="3" width="14" height="19" rx="2" fill="white"/><path d="M8 8h8M8 12h8M8 16h6" stroke="#e8794c" stroke-width="1.5"/>',
    ],
  };
  function icon(id) {
    const el = node("span", "app-icon");
    const app = apps[id];
    el.title = app ? app.name : id;
    el.setAttribute("aria-label", el.title);
    const mark = marks[id];
    if (mark) {
      el.style.background = mark[0];
      el.innerHTML =
        '<svg viewBox="0 0 24 24" aria-hidden="true">' + mark[1] + "</svg>";
    } else {
      el.textContent = id === "NO_LAUNCH" ? "—" : "…";
    }
    return el;
  }
  const sample = () =>
    data.samples.find(
      (r) =>
        r.user_id === state.user &&
        r.case === state.scenario &&
        r.variant === state.variant &&
        (!isTravelFocus() ||
          (r.input.last_app_id === state.travelLastApp &&
            r.input.calendar.starts_in_min === state.travelMinutes))
    );
  const isTravelFocus = () =>
    state.user === "u03" &&
    state.scenario === "map_trip" &&
    state.variant === "enroute";
  function choose(update) {
    Object.assign(state, update, { live: null });
    state.generation++;
    $("rawOutput").value = "";
    status("입력 선택을 바꿨습니다. 새 입력을 폰에 넣어 실행하세요.");
    render();
  }
  function choiceButton(label, selected, action) {
    const b = node("button", "", label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(selected));
    b.onclick = action;
    return b;
  }
  function fact(label, value) {
    $("contextFacts").append(node("dt", "", label), node("dd", "", value));
  }
  function render() {
    const r = sample(),
      input = r.input,
      msg = input.notification,
      cal = input.calendar;
    $("profiles").replaceChildren(
      ...Object.entries(profiles).map(([id, p]) =>
        choiceButton(p[0], id === state.user, () => choose({ user: id }))
      )
    );
    $("profileDescription").textContent = profiles[state.user][1];
    $("scenarios").replaceChildren(
      ...Object.entries(cases).map(([id, c], index) => {
        const b = choiceButton(c[0], id === state.scenario, () =>
          choose({ scenario: id, variant: Object.keys(c[1])[0] })
        );
        b.prepend(node("small", "", "SCENARIO 0" + (index + 1)));
        return b;
      })
    );
    $("variants").replaceChildren(
      ...Object.entries(cases[state.scenario][1]).map(([id, label]) =>
        choiceButton(label, id === state.variant, () => choose({ variant: id }))
      )
    );
    const focus = isTravelFocus();
    $("travelControls").hidden = !focus;
    $("pairFocus").hidden = !focus;
    $("travelLastApps").replaceChildren(
      ...["kakao_map", "youtube"].map((id) =>
        choiceButton("직전 " + apps[id].name, id === state.travelLastApp, () =>
          choose({ travelLastApp: id })
        )
      )
    );
    $("travelTimes").replaceChildren(
      ...[5, 8].map((minutes) =>
        choiceButton(
          "약속 " + minutes + "분 전",
          minutes === state.travelMinutes,
          () => choose({ travelMinutes: minutes })
        )
      )
    );
    const headlines = {
      map_trip: {
        departing: "약속 장소로 출발 전",
        enroute: "약속 장소로 이동 중",
        arrived: "약속 장소 도착 후",
        plan: "나중 약속 확인",
      },
      payment_request: {
        now: "즉시 정산 요청",
        later: "나중에 처리할 정산 요청",
        done: "정산 완료",
        account: "계좌 회신 요청",
      },
      contact_request: {
        family_now: "가족의 즉시 정산 요청",
        work_now: "동료의 즉시 정산 요청",
        family_later: "가족의 나중 정산 요청",
        work_later: "동료의 나중 정산 요청",
        family_done: "가족 정산 완료",
        work_done: "동료 정산 완료",
        family_account: "가족의 계좌 회신 요청",
        work_account: "동료의 계좌 회신 요청",
      },
      meeting_lifecycle: {
        upcoming: "회의 시작 전",
        join: "회의 참여",
        ended: "회의 종료 후",
        cancelled: "회의 취소",
      },
    };
    $("contextHeadline").textContent = headlines[state.scenario][state.variant];
    $("contextFacts").replaceChildren();
    fact("이동 상태", tr(input.journey));
    fact(
      "일정",
      (cal.kind === "none" ? "없음" : cal.kind === "travel" ? "약속" : "회의") +
        " · " +
        tr(cal.status) +
        (cal.kind !== "none" && cal.starts_in_min != null
          ? " · " + cal.starts_in_min + "분 뒤"
          : cal.kind !== "none" && cal.ends_ago_min != null
          ? " · " + cal.ends_ago_min + "분 전 종료"
          : "")
    );
    if (msg.kind === "payment") {
      fact("정산 상대", tr(msg.relationship));
      fact(
        "요청 · 처리",
        tr(msg.request_timing) + " · " + tr(msg.observed_status)
      );
    }
    const recent = [input.last_app_id];
    fact("직전 앱", apps[input.last_app_id].name);
    $("recentApps").replaceChildren(
      ...recent.map((id, index) => {
        const el = icon(id);
        if (index === recent.length - 1) el.classList.add("last-app");
        return el;
      })
    );
    $("rawPrompt").value = r.prompt;
    $("sampleMeta").textContent =
      r.id +
      " · compact test 예시 · " +
      r.input_length +
      " tokens · 선택한 입력의 정답은 폰에 보내지 않습니다.";
    renderResults();
  }

  function ranked(value) {
    if (value.probs) {
      return data.candidates
        .filter((a) => a.displayable)
        .sort(
          (a, b) =>
            value.probs[b.index] - value.probs[a.index] ||
            (a.app_id < b.app_id ? -1 : 1)
        )
        .slice(0, 4)
        .map((a) => ({ id: a.app_id, probability: value.probs[a.index] }));
    }
    return value.ids.map((id) => ({ id }));
  }
  function answerApps(r) {
    return [...new Set([r.target_app_id, ...r.target_app_ids])];
  }
  function recommendationScreen() {
    const screen = node("div", "recommendation-screen");
    screen.setAttribute("aria-label", "최근 앱 화면의 하단 추천 영역 예시");
    const snapshot = node("img", "recents-snapshot");
    snapshot.src = "assets/one_ui_recents.jpg";
    snapshot.alt = "";
    snapshot.setAttribute("aria-hidden", "true");
    const obscure = node("div", "recents-obscure");
    obscure.setAttribute("aria-hidden", "true");
    const dock = node("div", "recommendation-dock");
    screen.append(snapshot, obscure, dock);
    return { screen, dock };
  }
  function fillDock(dock) {
    while (dock.children.length < 4) {
      const empty = node("div", "dock-empty");
      empty.setAttribute("aria-hidden", "true");
      dock.append(empty);
    }
  }
  function renderAnswer(r, col) {
    col.append(node("span", "answer-badge", "정답 예시"));
    if (!r.reference_available)
      col.append(
        node("p", "answer-caption", "A는 미학습 · 데이터의 정답을 표시합니다.")
      );
    const { screen, dock } = recommendationScreen();
    for (const id of answerApps(r)) {
      const primary = id === r.target_app_id;
      const row = node(
        "div",
        "app-result answer-app" + (primary ? " answer-primary" : "")
      );
      row.dataset.appId = id;
      row.dataset.source = "dataset-answer";
      row.append(icon(id));
      const content = node("div", "answer-content");
      content.append(
        node("strong", "", apps[id].name),
        node("span", "", primary ? "정답" : "함께 추천")
      );
      row.append(content, node("span", "answer-check", "✓"));
      dock.append(row);
    }
    fillDock(dock);
    col.append(screen);
  }
  function renderResults() {
    const r = sample(),
      answerMode = state.resultMode === "answer",
      live = answerMode ? null : state.live;
    $("resultModes").replaceChildren(
      ...[
        ["answer", "정답 예시"],
        ["model", "모델 출력"],
      ].map(([mode, label]) =>
        choiceButton(label, state.resultMode === mode, () => {
          state.resultMode = mode;
          renderResults();
        })
      )
    );
    $("sourceBadge").textContent = answerMode
      ? "시나리오 정답 예시"
      : live
      ? live.source
      : r.reference_available
      ? "저장된 Python 추론"
      : "A 미학습 · 참조 추론 없음";
    $("outputDescription").textContent = answerMode
      ? "BEFORE: 저장 예측 · AFTER: 정답 예시"
      : live
      ? "현재 입력에 연결한 출력"
      : "동일 입력 · 학습 전후 비교";
    $("showReference").hidden = !live;
    $("sourceNote").textContent = answerMode
      ? "One UI 스크린샷을 배경으로 구성한 추천 화면 예시입니다. AFTER는 데이터의 정답·동반 추천, BEFORE는 저장된 Python 추론입니다."
      : "One UI 스크린샷을 배경으로 구성한 추천 화면 예시입니다. 기본 전후 확률은 저장된 Python 추론이며, 폰 결과를 읽으면 해당 출력으로 바뀝니다.";
    $("targetNote").textContent = answerMode
      ? "정답: " +
        apps[r.target_app_id].name +
        (r.target_app_ids.some((id) => id !== r.target_app_id)
          ? " · 함께 추천: " +
            r.target_app_ids
              .filter((id) => id !== r.target_app_id)
              .map((id) => apps[id].name)
              .join(", ")
          : "")
      : isTravelFocus()
      ? "학습 목표: 카카오맵과 YouTube 모두 Top4에 포함 · 1위/교대 순서는 평가 기준이 아닙니다."
      : "학습 목표: " +
        apps[r.target_app_id].name +
        (r.target_app_ids.length > 1
          ? " 우선 · " +
            r.target_app_ids
              .filter((id) => id !== r.target_app_id)
              .map((id) => apps[id].name)
              .join(", ") +
            "도 추천에 포함"
          : "");
    $("predictionResults").replaceChildren(
      ...["before", "after"].map((key) => {
        const col = node("div", "result-column " + key);
        col.append(
          node(
            "p",
            "result-heading " + key,
            key === "before" ? "BEFORE · 기본 모델" : "AFTER · 개인화 학습 후"
          )
        );
        if (key === "after" && answerMode) {
          renderAnswer(r, col);
          return col;
        }
        const value = live ? live[key] : r[key];
        if (!value) {
          col.append(
            node(
              "p",
              "empty-result",
              !live && !r.reference_available
                ? "A는 데이터만 준비되어 있습니다. 저장된 모델 예측이 없습니다."
                : "아직 읽은 출력이 없습니다."
            )
          );
          return col;
        }
        col.append(
          node(
            "span",
            "screen-source",
            live ? live.source : "저장된 Python 추론"
          )
        );
        if (value.probs) {
          const top = value.probs.indexOf(Math.max(...value.probs));
          if (!data.candidates[top].displayable)
            col.append(
              node("p", "subtle", "최상위 판단: " + data.candidates[top].name)
            );
        }
        const { screen, dock } = recommendationScreen();
        for (const [index, app] of ranked(value).entries()) {
          const row = node("div", "app-result");
          row.dataset.appId = app.id;
          if (isTravelFocus() && ["kakao_map", "youtube"].includes(app.id))
            row.classList.add("pair-target");
          row.append(icon(app.id));
          const content = node("div"),
            line = node("div", "app-result-name");
          const label = node("strong", "", apps[app.id].name);
          label.title = label.textContent;
          line.append(
            label,
            node(
              "span",
              "",
              app.probability == null
                ? "#" + (index + 1)
                : app.probability > 0 && app.probability < 0.001
                ? "<0.1%"
                : (100 * app.probability).toFixed(1) + "%"
            )
          );
          content.append(line);
          row.append(content);
          dock.append(row);
        }
        fillDock(dock);
        col.append(screen);
        return col;
      })
    );
    const after = live ? live.after : r.after;
    if (isTravelFocus()) {
      if (answerMode) {
        $("pairStatus").textContent = "정답 예시 · 두 앱 선택";
        $("pairStatus").classList.remove("pair-passed");
        return;
      }
      const recommended = after ? ranked(after).map((app) => app.id) : [];
      const count = ["kakao_map", "youtube"].filter((id) =>
        recommended.includes(id)
      ).length;
      $("pairStatus").textContent = !after
        ? "AFTER 출력 대기"
        : count === 2
        ? "AFTER · 두 앱 모두 포함"
        : recommended.length < 4
        ? "출력 " + recommended.length + "개 · 동시 Top4 확인 전"
        : "AFTER · 목표 앱 " + count + "/2 포함";
      $("pairStatus").classList.toggle("pair-passed", count === 2);
    }
  }

  function resolveApp(value) {
    if (typeof value !== "string") throw Error("앱 이름 또는 ID가 필요합니다.");
    const text = value.trim();
    const app = data.candidates.find(
      (a) => a.app_id === text || a.name === text || a.letter === text
    );
    if (!app) throw Error("후보 목록에 없는 출력입니다: " + text.slice(0, 80));
    return app.app_id;
  }
  function parseOutput(raw) {
    let value;
    const text = raw
      .trim()
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/, "");
    if (!text) throw Error("출력이 비어 있습니다.");
    try {
      value = JSON.parse(text);
    } catch {
      value = text;
    }
    if (value && !Array.isArray(value) && typeof value === "object") {
      if (value.probs != null) {
        const p = value.probs;
        if (
          !Array.isArray(p) ||
          p.length !== 14 ||
          p.some(
            (x) =>
              typeof x !== "number" || !Number.isFinite(x) || x < 0 || x > 1
          ) ||
          Math.abs(p.reduce((a, b) => a + b, 0) - 1) > 1e-4
        )
          throw Error("probs는 합이 1인 14개 확률이어야 합니다.");
        return { probs: p };
      }
      value =
        value.recommended_app_ids ??
        value.predicted_app_id ??
        value.app_id ??
        value.answer;
      if (value == null)
        throw Error(
          "probs, recommended_app_ids, predicted_app_id 또는 answer가 필요합니다."
        );
    }
    // Do not scan arbitrary prose: a prompt echo or a log is not a model decision.
    const ids = Array.isArray(value)
      ? value.map(resolveApp)
      : [resolveApp(value)];
    if (ids.length < 1 || ids.length > 4 || new Set(ids).size !== ids.length)
      throw Error("중복 없이 최대 4개 앱을 지정하세요.");
    return { ids };
  }
  function status(message, error = false) {
    $("actionStatus").textContent = message;
    $("actionStatus").classList.toggle("error", error);
  }
  let token;
  async function api(action, body = {}) {
    if (!token) {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok)
        throw Error(
          "기기 연결 서버가 없습니다. scripts/serve_app_demo.py로 8001 서버를 실행하세요."
        );
      token = (await response.json()).token;
    }
    const response = await fetch("/api/" + action, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Demo-Token": token },
      body: JSON.stringify(body),
    });
    const value = await response.json();
    if (!response.ok) throw Error(value.error || "기기 요청이 실패했습니다.");
    return value;
  }
  async function phoneAction(action) {
    const buttons = ["openPhone", "sendPhone", "readPhone"].map($);
    buttons.forEach((b) => (b.disabled = true));
    const id = sample().id,
      generation = state.generation;
    status("휴대폰과 연결하고 있습니다…");
    try {
      const result = await api(action, { sample_id: id, udid });
      if (generation !== state.generation) {
        status("선택이 바뀌었습니다. 새 입력으로 다시 연결하세요.");
        return;
      }
      if (action === "read") {
        const parsed = { source: "기기 화면에서 읽은 출력" };
        const errors = [];
        for (const key of ["before", "after"])
          if (result[key]) {
            try {
              parsed[key] = parseOutput(result[key]);
            } catch (e) {
              errors.push(key + ": " + e.message);
            }
          }
        if (!parsed.before && !parsed.after)
          throw Error(
            errors.join(" / ") ||
              "아직 앱 선택 출력이 없습니다. 폰에서 BEFORE 또는 AFTER를 실행하세요."
          );
        state.live = parsed;
        state.resultMode = "model";
        renderResults();
        status(
          "기기 출력 표시 · " +
            result.captured_at +
            (errors.length ? " · " + errors.join(" / ") : "") +
            " · 폰 모델/adapter 설정을 기준으로 합니다."
        );
      } else {
        status(result.message);
      }
    } catch (e) {
      status(e.message, true);
    } finally {
      buttons.forEach((b) => (b.disabled = false));
    }
  }
  $("copyPrompt").onclick = async () => {
    try {
      await navigator.clipboard.writeText(sample().prompt);
      status("정답을 제외한 모델 입력을 복사했습니다.");
    } catch {
      $("inputDetails").open = true;
      $("rawPrompt").select();
      status(
        "브라우저 클립보드 접근이 제한되어 있습니다. 선택된 입력을 Ctrl+C로 복사하세요."
      );
    }
  };
  $("openPhone").onclick = () => phoneAction("open");
  $("sendPhone").onclick = () => phoneAction("prepare");
  $("readPhone").onclick = () => phoneAction("read");
  $("showReference").onclick = () => {
    state.live = null;
    state.resultMode = "model";
    renderResults();
    status("저장된 Python 추론을 표시합니다.");
  };
  $("parseOutput").onclick = () => {
    try {
      const value = parseOutput($("rawOutput").value);
      if (state.live?.source !== "직접 붙여넣은 출력")
        state.live = { source: "직접 붙여넣은 출력" };
      state.live[$("outputColumn").value] = value;
      state.resultMode = "model";
      renderResults();
      status(
        "붙여넣은 출력을 앱으로 표시했습니다. 확률이 없는 출력은 순위만 표시합니다."
      );
    } catch (e) {
      status(e.message, true);
    }
  };
  $("expandDiagram").onclick = () => $("diagramDialog").showModal();
  $("closeDiagram").onclick = () => $("diagramDialog").close();
  render();
  // Expose only pure parsers and selection helpers for the local UI checks.
  window.Demo = { parseOutput, choose, sample, state };
})();
