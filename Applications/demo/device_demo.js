"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const ids = [
    "naver_map",
    "kakao_map",
    "kakao_t",
    "kakaotalk",
    "toss",
    "kakaopay",
    "youtube",
    "calendar_app",
    "meet",
    "ibk_bank",
    "anki",
    "notes",
    "OTHER_APP",
    "NO_LAUNCH",
  ];
  const names = [
    "네이버지도",
    "카카오맵",
    "카카오T",
    "카카오톡",
    "토스",
    "카카오페이",
    "YouTube",
    "Calendar",
    "Meet",
    "기업은행",
    "Anki",
    "노트",
    "기타",
    "미사용",
  ];
  // 사용 빈도 기준 추천 (temporarily hidden; see the comment in demo_combined.html).
  const SHOW_FREQUENCY = false;
  let device = null,
    icons = {},
    token,
    connected = false,
    sending = false;
  let lastPhoneUser = null;
  let selectedUser = "u03",
    selectedScenario = 0,
    serial = "";
  const element = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text != null) el.textContent = text;
    return el;
  };
  const user = () => device?.users.find((row) => row.id === selectedUser);
  const scenario = () => user()?.scenarios[selectedScenario];
  const visibleByUser = {};
  function visibleScenarios(u) {
    return visibleByUser[u.id] ||= u.scenarios.map((sc, index) => ({label: sc.title, index, featured: sc.featured}))
      .filter(row => row.featured).slice(0, 4);
  }
  function lastAppIndex(sc) {
    return ids.indexOf(sc.observed_input.recent_apps.at(-1));
  }
  const pct = (value) => (100 * value).toFixed(1) + "%";
  const number = (value, digits = 2) =>
    Number.isFinite(value) ? value.toFixed(digits) : "—";
  function top4(result) {
    if (!result) return [];
    return Array.from({ length: 12 }, (_, index) => index)
      .sort((a, b) => result.probs[b] - result.probs[a] || a - b)
      .slice(0, 4);
  }
  function icon(index) {
    const el = element("span", "app-icon");
    const img = element("img");
    img.src = icons[ids[index]]?.asset || "";
    img.alt = names[index];
    el.append(img);
    return el;
  }
  function button(label, selected, action) {
    const el = element("button", "", label);
    el.type = "button";
    el.setAttribute("aria-pressed", String(selected));
    el.onclick = action;
    return el;
  }
  function mirror() {
    if (!serial) return;
    const params = new URLSearchParams(location.search);
    const url = new URL(params.get("server") || "http://localhost:8002/");
    const ws = new URL(url);
    ws.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    ws.search = new URLSearchParams({
      action: "proxy-adb",
      remote: "tcp:8886",
      udid: serial,
    });
    url.hash =
      "!" +
      new URLSearchParams({
        action: "stream",
        udid: serial,
        player: "mse",
        fitToScreen: "true",
        ws: ws.toString(),
      });
    if ($("mirror").src !== url.toString()) $("mirror").src = url;
  }
  async function request(path, body) {
    const response = await fetch(path, {
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
      ...(body
        ? {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-Demo-Token": token,
            },
            body: JSON.stringify(body),
          }
        : {}),
    });
    const value = await response.json();
    if (!response.ok) throw Error(value.error || "기기 연결 실패");
    return value;
  }
  function accept(value) {
    const next = value.device;
    if (!next || next.source !== "android_native" || next.schema !== 1)
      throw Error("실제 기기 출력 형식이 아닙니다.");
    if (next.dataset_id !== "personalization_bc_observed_v1" || next.run_all_mode !== "train_and_infer_each_user" ||
        next.app_catalog_sha256 !== "04450d4822f9ac8e821a1e5d006ae9d8b28fdeef3bdf8bd759faa4a99f7a54a9" ||
        !next.app_catalog?.every((row, i) => row.id === ids[i] && row.name === names[i]))
      throw Error("새 B/C 관측 데이터용 Quick.AI 앱과 후보 매핑을 확인하세요.");
    for (const row of next.users) {
      if (row.data_rows < 200 || row.data_rows !== row.expected_data_rows || row.data_sha256 !== row.expected_data_sha256 ||
          !row.scenarios.length || row.scenarios.length > 4 || row.validation_rows !== 80 || row.test_rows !== 80)
        throw Error("기기 데이터가 B/C 학습·검증 데이터와 일치하지 않습니다.");
      const frequency = row.frequency_baseline;
      if (frequency?.source !== "training_answer_counts" || frequency.total !== row.data_rows ||
          frequency.train_sha256 !== row.data_sha256 || frequency.counts.length !== 14 ||
          frequency.counts.reduce((a, b) => a + b, 0) !== row.data_rows)
        throw Error("학습 데이터의 사용 빈도 출처를 확인할 수 없습니다.");
      for (const sc of row.scenarios) {
        if (Object.keys(sc.observed_input || {}).sort().join() !== "calendar,moving,notification,recent_apps,time" ||
            sc.observed_input.recent_apps.length !== 4)
          throw Error("기기의 다섯 관측 입력을 확인할 수 없습니다.");
      }
    }
    if (
      device?.session_id === next.session_id &&
      next.sequence < device.sequence
    )
      return;
    if (connected && device?.session_id === next.session_id && next.sequence === device.sequence)
      return;
    for (const u of next.users)
      for (const sc of u.scenarios)
        for (const phase of ["before", "after"]) {
          const result = sc[phase];
          if (
            result &&
            (result.source !== "android_native" ||
              result.prompt_sha256 !== sc.prompt_sha256 ||
              result.probs.length !== 14 ||
              result.probs.some((p) => !Number.isFinite(p) || p < 0 || p > 1) ||
              Math.abs(result.probs.reduce((a, b) => a + b, 0) - 1) > 0.002)
          )
            throw Error("기기 확률/입력 연결을 확인할 수 없습니다.");
        }
    next.users = next.users.filter((u) => ["B", "C"].includes(u.label));
    // Follow the user picked (or being run) on the phone.
    const phoneUser = next.busy ? next.active_user : next.selected_user;
    if (phoneUser && phoneUser !== lastPhoneUser && next.users.some((u) => u.id === phoneUser)) {
      if (phoneUser !== selectedUser) {
        selectedUser = phoneUser;
        selectedScenario = visibleScenarios(next.users.find((u) => u.id === phoneUser))[0]?.index || 0;
      }
      lastPhoneUser = phoneUser;
    }
    device = next;
    connected = true;
    serial = value.serial;
    mirror();
    render();
  }
  async function poll() {
    try {
      accept(await request("/api/device"));
    } catch (error) {
      connected = false;
      $("sourceBadge").textContent = device
        ? "연결 끊김 · 마지막 수신 결과"
        : "기기 연결 대기";
      $("deviceConnection").textContent = error.message;
      controls();
    }
    setTimeout(poll, 1200);
  }
  async function command(action, targetId = selectedUser) {
    if (sending) return;
    sending = true;
    controls();
    $("actionStatus").textContent = "기기에 요청을 보내고 있습니다…";
    try {
      // Bind the requested user before any asynchronous work. Native training
      // changes only this user's adapter; select also focuses the phone chart.
      if (["train", "after", "run_user"].includes(action)) {
        if (selectedUser !== targetId) {
          selectedUser = targetId;
          selectedScenario = visibleScenarios(user())[0]?.index || 0;
          render();
        }
        accept(await request("/api/command", {
          action: "select", user: targetId, request_id: crypto.randomUUID(),
        }));
      }
      const value = await request("/api/command", {
        action,
        user: targetId,
        request_id: crypto.randomUUID(),
      });
      accept(value);
      $("actionStatus").textContent =
        "기기가 요청을 받았습니다. 진행 상태와 결과를 자동으로 갱신합니다.";
    } catch (error) {
      $("actionStatus").textContent = error.message;
    } finally {
      sending = false;
      controls();
    }
  }
  function controls() {
    const busy = Boolean(device?.busy || sending),
      u = user();
    $("runUser").disabled = !connected || busy;
    $("runUser").textContent = `사용자 ${u?.label || "C"} 추론 실행 · 추가 학습 없음`;
  }
  // One dock cell; shared by the phone mockups and the frequency strip so both
  // use the same geometry and the same target outline.
  function dockItem(index, value, sc) {
    const row = element("div", "app-result");
    row.dataset.appId = ids[index];
    if (expectedApps(sc).includes(index)) row.classList.add("native-target");
    row.append(icon(index));
    const text = element("div", "app-result-name");
    text.append(element("strong", "", names[index]), element("span", "", value));
    row.append(text);
    return row;
  }
  const expectedApps = (sc) => sc.required_indices.length ? sc.required_indices : [sc.answer_index];
  const personas = {
    B: {title: "개인사업자", summary: "길찾기는 네이버지도, 송금은 사업용과 개인용을 나눠 씁니다.",
      facts: [["이동", 0], ["사업 송금", 9], ["개인 송금", 5]]},
    C: {title: "대학생", summary: "이동할 땐 카카오맵과 YouTube를 함께 켜고, 송금은 모두 토스로 합니다.",
      facts: [["이동", 1], ["영상", 6], ["모든 송금", 4]]},
  };
  // Why each user's labelled answer follows from the persona, per scenario variant.
  const reasons = {
    B: {moving_map: "이동 중 길찾기는 항상 네이버지도", moving_video: "직전에 영상을 봤어도 이동 중엔 네이버지도",
      business_goods: "물품 대금은 사업 송금 → 기업은행", personal_meal: "점심값은 개인 송금 → 카카오페이",
      moving_appointment: "방금 카카오맵을 봤어도, 길찾기는 항상 네이버지도"},
    C: {moving_map: "이동 중엔 카카오맵 길찾기와 YouTube를 함께", moving_video: "보던 YouTube를 이어 보며 카카오맵으로 이동",
      business_goods: "용도와 상관없이 송금은 토스", personal_meal: "개인 송금도 토스",
      moving_appointment: "카카오맵으로 길 확인 후, 이동 중엔 YouTube"},
  };
  function selectUser(row) {
    if (row.id === selectedUser) return;
    selectedUser = row.id;
    selectedScenario = visibleScenarios(row)[0]?.index || 0;
    render();
    command("select");
  }
  function phone(result, label, sc) {
    const col = element("div", "result-column " + label);
    const shell = element("div", "phone-shell");
    const screen = element("div", "recommendation-screen");
    // Reference One UI chrome, independent of native model measurements.
    const status = element("div", "phone-statusbar");
    status.setAttribute("aria-hidden", "true");
    const statusLeft = element("span", "status-left");
    statusLeft.append(element("span", "status-time", sc.observed_input.time), element("span", "status-notification"));
    const statusRight = element("span", "status-right");
    const signal = element("span", "status-signal");
    for (let i = 0; i < 4; i++) signal.append(element("i"));
    statusRight.append(element("span", "status-network", "LTE"), signal, element("span", "status-battery", "92"));
    status.append(statusLeft, statusRight);
    screen.append(status);
    const last = lastAppIndex(sc);
    // Neutral previous window: decoration, not a fabricated app-history entry.
    const previousWindow = element("div", "recent-app-backdrop");
    previousWindow.setAttribute("aria-hidden", "true");
    previousWindow.append(element("div", "backdrop-toolbar"));
    for (let i = 0; i < 3; i++)
      previousWindow.append(element("div", "backdrop-content"));
    const recentCard = element("div", "recent-app-card");
    recentCard.dataset.appId = ids[last] || "unknown";
    recentCard.setAttribute("aria-label", "직전 사용 앱: " + (names[last] || "없음"));
    const badge = element("div", "recent-app-badge");
    if (last >= 0) badge.append(icon(last));
    recentCard.append(badge, element("div", "recent-app-window"));
    const closeAll = element("div", "recents-close-all", "모두 닫기");
    closeAll.setAttribute("aria-hidden", "true");
    screen.append(previousWindow, recentCard, closeAll);
    if (!result)
      screen.append(
        element(
          "p",
          "phone-wait",
          label === "before"
            ? "학습 전 추론을 실행하세요"
            : "학습 후 추론을 실행하세요"
        )
      );
    const dock = element("div", "recommendation-dock");
    for (const index of top4(result))
      dock.append(dockItem(index, pct(result.probs[index]), sc));
    while (dock.children.length < 4) dock.append(element("div", "dock-empty"));
    screen.append(dock);
    shell.append(screen);
    const frame = element("img", "phone-frame");
    frame.src = "assets/galaxy_s26_ultra_white.png";
    frame.alt = "";
    shell.append(frame);
    const heading = element("p", "result-heading " + label);
    heading.append(element("strong", "", label === "before" ? "학습 전" : "개인화 학습 후"),
      element("span", "", label === "before" ? "BEFORE" : "AFTER"));
    col.append(shell, heading);
    return col;
  }
  function renderPreferences(u, sc) {
    $("userPreferences").replaceChildren(...device.users.map(row => {
      const label = row.label, persona = personas[label], active = row.id === u.id;
      const card = element("div", "user-overview preference-card" + (active ? " selected" : ""));
      card.dataset.user = label;
      card.setAttribute("role", "button");
      card.setAttribute("aria-pressed", String(active));
      card.tabIndex = 0;
      card.onclick = () => selectUser(row);
      card.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectUser(row); }
      };
      const heading = element("header", "persona-header");
      const identity = element("div", "persona-identity");
      identity.append(element("span", "persona-user", `사용자 ${label}`), element("h3", "persona-title", persona.title));
      heading.append(element("span", "user-avatar", label), identity);
      heading.append(element("span", "persona-selected", active ? "선택 중" : "선택"));
      const facts = element("dl", "persona-patterns");
      for (const [category, app] of persona.facts) {
        const item = element("div", "persona-pattern"), value = element("dd");
        value.append(icon(app), element("span", "", names[app]));
        item.append(element("dt", "", category), value);
        facts.append(item);
      }
      card.append(heading, element("p", "persona-summary", persona.summary), facts);
      return card;
    }));
    renderExpectation(u, sc);
  }
  function appChip(index, cls = "expect-app") {
    const app = element("span", cls);
    app.append(icon(index), element("span", "", names[index]));
    return app;
  }
  function renderExpectation(u, sc) {
    const expected = expectedApps(sc);
    const reason = element("div", "expect-reason");
    reason.append(element("span", "user-avatar", u.label),
      element("p", "", `${personas[u.label].title} · ${reasons[u.label]?.[sc.variant] || "학습 데이터의 선호 패턴"}`));
    const apps = element("div", "expect-apps");
    apps.append(...expected.map(index => appChip(index)));
    const main = element("div", "expect-main"), line = element("div", "expect-line");
    line.append(apps);
    main.append(reason, line);

    const checks = element("dl", "expect-checks");
    const freq = u.frequency_baseline.top4.map(row => row.index);
    const rows = [["BEFORE", sc.before && top4(sc.before)], ["빈도 추천", freq], ["AFTER", sc.after && top4(sc.after)]]
      .filter(([name]) => SHOW_FREQUENCY || name !== "빈도 추천");
    for (const [name, top] of rows) {
      const hit = top ? expected.filter(i => top.includes(i)).length : null;
      const value = element("dd", hit == null ? "pending" : hit === expected.length ? "hit" : "miss",
        hit == null ? "대기" : `${hit === expected.length ? "✓" : "✗"} ${hit}/${expected.length}`);
      checks.append(element("dt", "", name), value);
    }
    const side = element("div", "expect-side");
    side.append(element("div", "expect-side-label", "Top4 포함"), checks);
    const peer = device.users.find(row => row.id !== u.id);
    const peerSc = peer?.scenarios.find(row => row.event_id === sc.event_id);
    if (peerSc) {
      const contrast = element("div", "expect-contrast");
      contrast.append(element("span", "", `같은 입력, 사용자 ${peer.label}라면`));
      const peerApps = element("span", "expect-contrast-apps");
      peerApps.append(...expectedApps(peerSc).map(index => appChip(index, "expect-contrast-app")));
      contrast.append(peerApps);
      line.append(contrast);
    }
    $("expectation").replaceChildren(main, side);
  }
  function renderTrainingDetails(u) {
    const epochs = u.epochs;
    $("lossChart").replaceChildren();
    if (u.iterations.length) {
      const ns = "http://www.w3.org/2000/svg",
        svg = document.createElementNS(ns, "svg");
      svg.setAttribute("viewBox", "0 0 720 180");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "기기에서 받은 iteration별 training loss");
      const values = u.iterations.map((r) => r.loss),
        max = Math.max(...values),
        min = Math.min(...values);
      const polyline = document.createElementNS(ns, "polyline");
      polyline.setAttribute(
        "points",
        values
          .map(
            (v, i) =>
              `${18 + (i * 684) / Math.max(values.length - 1, 1)},${
                160 - ((v - min) * 136) / Math.max(max - min, 0.01)
              }`
          )
          .join(" ")
      );
      polyline.setAttribute("fill", "none");
      polyline.setAttribute("stroke", "var(--green)");
      polyline.setAttribute("stroke-width", "2.5");
      svg.append(polyline);
      $("lossChart").append(svg);
      const table = element("table", "epoch-table"),
        head = element("tr");
      for (const label of ["Epoch", "Train loss", "평가 loss", "학습 시간"])
        head.append(element("th", "", label));
      table.append(head);
      for (const epoch of epochs) {
        const row = element("tr");
        for (const val of [
          epoch.epoch,
          number(epoch.train_loss, 4),
          number(epoch.valid_loss, 4),
          number(epoch.train_seconds, 3) + " s",
        ])
          row.append(element("td", "", String(val)));
        table.append(row);
      }
      $("lossChart").append(table);
    } else
      $("lossChart").append(
        element(
          "p",
          "subtle",
          "학습을 실행하면 기기의 loss와 epoch별 통계를 표시합니다."
        )
      );
    $("trainingConfig").textContent = u.train_config
      ? `rank ${u.train_config.lora_rank} · 마지막 ${u.train_config.lora_last_n} blocks · ${u.train_config.epochs}epochs · batch ${u.train_config.batch_size} · seed ${u.train_config.seed} · 학습 데이터 ${u.data_rows}개 · ${u.data_path}`
      : `학습 데이터 ${u.data_rows}개 · ${u.data_path}`;
  }
  function render() {
    const u = user();
    if (!u) return;
    if (selectedScenario >= u.scenarios.length) selectedScenario = 0;
    const sc = scenario();
    $("scenarios").replaceChildren(
      ...visibleScenarios(u).map(({label, index}) =>
        button(label, index === selectedScenario, () => {
          selectedScenario = index;
          render();
        })
      )
    );
    $("chooseScenario").hidden = u.scenarios.length <= 4;
    $("scenarioChoices").replaceChildren(...u.scenarios.map((row, index) => {
      const el = button(row.title + " · 정답 " + names[row.answer_index], index === selectedScenario, () => {
        if (!visibleScenarios(u).some((r) => r.index === index))
          visibleByUser[u.id][3] = {label: row.title, index};
        selectedScenario = index;
        render();
        $("scenarioDialog").close();
      });
      const input = row.observed_input;
      el.append(element("small", "", input.notification || input.calendar?.title || "알림·일정 없음"));
      return el;
    }));
    for (const id of ["variants", "travelControls"]) $(id).hidden = true;
    $("sourceBadge").textContent = connected
      ? "기기 실측"
      : "연결 끊김 · 마지막 수신 결과";
    $(
      "deviceConnection"
    ).textContent = `${device.device_model} · Android ${device.android_version} · ${serial}`;
    $("deviceProgress").textContent = device.message.replace(/\s*·\s*loss\s+[-+\d.eE]+/g, "");
    $("deviceProgress").classList.toggle(
      "error",
      device.command_status === "error"
    );
    $("contextHeadline").textContent = sc.situation;
    $("contextFacts").replaceChildren();
    const input = sc.observed_input;
    const facts = [["시간", input.time], ["이동", input.moving ? "이동 중" : "이동 없음"],
      ["일정", input.calendar ? `${input.calendar.start}–${input.calendar.end} ${input.calendar.title}` : "없음"],
      ["알림", input.notification || "없음"]];
    for (const [key, value] of facts) {
      const dd = element("dd", key === "알림" ? "notification-value" : "", value);
      $("contextFacts").append(element("dt", "", key), dd);
    }
    $("recentApps").replaceChildren(...input.recent_apps.flatMap((id, index) => {
      const app = ids.indexOf(id), item = element("span", "recent-app-item" + (index === 3 ? " recent-last" : ""));
      item.append(icon(app), element("span", "", names[app]));
      return index ? [element("span", "recent-arrow", "›"), item] : [item];
    }));
    const frequency = u.frequency_baseline;
    if (SHOW_FREQUENCY) $("frequencyApps").replaceChildren(...Array.from({length: 4}, (_, rank) => {
      const row = frequency.top4[rank];
      if (!row) return element("div", "dock-empty");
      const item = dockItem(row.index, pct(row.share), sc);
      item.title = `${row.count}회 / ${frequency.total}건`;
      return item;
    }));
    renderPreferences(u, sc);
    $("resultModes").replaceChildren(
      element("span", "subtle", "폰에서 받은 Top4 · 추천 화면 하단")
    );
    $("predictionResults").replaceChildren(
      phone(sc.before, "before", sc),
      phone(sc.after, "after", sc)
    );
    $("targetNote").textContent =
      `기대 앱: ${expectedApps(sc).map(i => names[i]).join(" · ")} · Top4에 포함되면 테두리 표시`;
    const required = sc.required_indices;
    $("pairFocus").hidden = !required.length;
    if (required.length) {
      $("pairFocus").querySelector("strong").textContent = required
        .map((i) => names[i])
        .join(" + ");
      $("pairFocus").querySelector("div > span").textContent =
        "데이터에 지정된 동시 추천 목표";
      $("pairStatus").textContent = sc.after
        ? `AFTER · ${
            required.filter((i) => top4(sc.after).includes(i)).length
          }/${required.length} 포함`
        : "AFTER 출력 대기";
    }
    $("rawPrompt").value = sc.prompt;
    $("rawOutput").value = JSON.stringify(
      {
        dataset_id: device.dataset_id,
        app_catalog: device.app_catalog,
        observed_input: sc.observed_input,
        sample_id: sc.sample_id,
        split: sc.split,
        session_id: device.session_id,
        user: u.id,
        scenario: sc.id,
        prompt_sha256: sc.prompt_sha256,
        before: sc.before || null,
        after: sc.after || null,
        epochs: u.epochs,
        train_config: u.train_config,
        adapter_path: u.adapter_path,
        adapter_sha256: u.adapter_sha256,
        training_wall_ms: u.training_wall_ms,
        data_sha256: u.data_sha256,
      },
      null,
      2
    );
    $(
      "sampleMeta"
    ).textContent = `기기 validation 시나리오 ${sc.id} · 입력 SHA256 ${sc.prompt_sha256}`;
    $(
      "sourceNote"
    ).textContent = `Android native API의 실제 출력입니다. 세션 ${device.session_id.slice(
      0,
      8
    )} · 수신 ${new Date(device.updated_at_ms).toLocaleTimeString(
      "ko-KR"
    )} · 앱 PSS ${number(device.process_pss_kb / 1024, 1)} MiB`;
    renderTrainingDetails(u);
    controls();
  }
  $("runUser").onclick = () => command("run_user");
  $("copyPrompt").onclick = async () => {
    try {
      await navigator.clipboard.writeText(scenario()?.prompt || "");
      $("actionStatus").textContent = "기기의 입력 원문을 복사했습니다.";
    } catch {
      $("inputDetails").open = true;
      $("rawPrompt").select();
    }
  };
  $("expandDiagram").onclick = () => $("diagramDialog").showModal();
  $("closeDiagram").onclick = () => $("diagramDialog").close();
  $("chooseScenario").onclick = () => $("scenarioDialog").showModal();
  $("closeScenarios").onclick = () => $("scenarioDialog").close();
  $("resetScenarios").onclick = () => {
    delete visibleByUser[selectedUser];
    selectedScenario = visibleScenarios(user())[0]?.index || 0;
    render();
    $("scenarioDialog").close();
  };
  $("closeDetails").onclick = () => $("detailsDialog").close();
  controls();
  mirror();
  Promise.all([
    request("/api/status"),
    fetch("assets/icons/sources.json").then((r) => r.json()),
  ])
    .then(([status, sources]) => {
      token = status.token;
      serial = status.serial;
      icons = sources.icons;
      mirror();
      poll();
    })
    .catch((error) => {
      $("actionStatus").textContent = error.message;
    });
  window.DeviceDemo = {
    top4,
    get state() {
      return device;
    },
    get selectedUser() {
      return selectedUser;
    },
  };
})();
