# Quick.AI B/C on-device personalization demo

`demo_combined.html` combines Android screen mirroring with four next-app scenarios,
before/after probabilities, per-user training controls and a training-frequency
baseline. Predictions and training statistics come from the Android app through
ADB. The page does not substitute saved Python predictions for device output.

## Included files

- `demo_combined.html`, `demo.css`, `device_demo.js`, local icons/fonts/diagrams.
- `tools/serve_device_demo.py`: Python standard-library HTTP/ADB bridge.
- `data/personalization_bc_observed_v1`: immutable original B/C observations,
  train200/validation80/test80 per person, schema and candidate catalog.
- `data/device/u02,u03`: pretokenized native JSONL inputs with one-hot targets.
- `data/demo-catalog.json` and `data/scenario-selection.json`: the four validation
  scenarios and their selection/provenance. J/index9 is **기업은행**, not Clock.
- Android source in
  `../CausalLM/Android/SampleTestAPP/src/main/java/com/example/sampletestapp/`:
  `MainActivity.kt`, `DeviceDemoBridge.kt`, `ObservedDemoCatalog.kt`.

The host server and deployment tool need Python3.10+ and Android platform-tools
(`adb` on PATH). They do not need PyTorch, Transformers or a host GPU.
The Android app needs an arm64 device, Android13/API33 or newer, the matching
NNTrainer native libraries and separately prepared native model files.

## Build and install the Android app

From the **nntrainer repository root**, use the existing native/app build workflow:

```sh
export ANDROID_NDK=/absolute/path/to/android-sdk/ndk/27.0.12077973
bash Applications/CausalLM/build_android.sh --app
```

Use the normal engine Meson options for your target GPU/OpenCL configuration;
this command alone does not certify a new device's hybrid GPU support. See
`../CausalLM/README_GPU.md` for the engine paths and backend requirements.

If you already have a matching, verified native snapshot, you can rebuild only
the UI without recompiling native libraries. The snapshot directory must contain
`arm64-v8a/*.so`, including `libquickai_jni.so`, `libquick_dot_ai_api.so`,
`libcausallm.so`, `libnntrainer.so`, `libccapi-nntrainer.so`, `libc++_shared.so`
and the LiteRT library used by the app:

```sh
cd Applications/CausalLM/Android
./gradlew -PnntrainerNdkPath=/absolute/path/to/android-sdk/ndk/27.0.12077973 \
  -PquickaiNativeSnapshot=/absolute/path/to/native-snapshot \
  :SampleTestAPP:assembleDebug
adb devices
adb -s YOUR_SERIAL install -r SampleTestAPP/build/outputs/apk/debug/SampleTestAPP-debug.apk
```

Set `ANDROID_HOME` and use JDK17 for the Android build. On Windows use
`gradlew.bat` and absolute Windows SDK/snapshot paths. Installing an APK built
with a different signing key cannot update the old app in place; use the same
key as your existing installation when preserving its data.

## Prepare the device data and models

From this directory (`Applications/demo`):

```sh
python tools/verify_bundle.py
python tools/deploy_device_data.py --serial YOUR_SERIAL --dry-run
python tools/deploy_device_data.py --serial YOUR_SERIAL
```

The deployment command uploads the six verified numeric data files and catalogs
to `/sdcard/quickai_bc_observed_v1`. It verifies each uploaded SHA256. It does not
start the app, train, infer, install an APK or copy old adapters.

The app expects two **NNTrainer-converted native model folders** at:

```text
/sdcard/quickai_bc_observed_v1/cpu/
/sdcard/quickai_bc_observed_v1/gpu/
```

Each folder contains `config.json`, `decision.json`, `generation_config.json`,
`nntr_config.json`, `tokenizer.json`, `tokenizer_config.json` and the binary named
by `nntr_config.json:model_file_name`. Use Qwen3-0.6B RLCD decision with the
original decision head/letter IDs and tokenizer. CPU training uses QS4CX/FP32
embedding/head; the hybrid trunk uses the corresponding QS4CX/Q6_K mixed GPU
model. A Hugging Face safetensors file or Python LoRA `.pt` file cannot replace
these native models.

To upload prepared model directories together with the data:

```sh
python tools/deploy_device_data.py --serial YOUR_SERIAL \
  --cpu-model-dir /path/to/prepared/cpu --gpu-model-dir /path/to/prepared/gpu
```

The tool uploads only the model/config/tokenizer whitelist, never adapter files.
Weights, APKs, trained adapters, SDK/NDK and Gradle caches are not part of this
commit. If models are already in the required device folders, omit both model
options. Both B and C use their own adapter selected by the app.

## Start the web/ADB bridge

```sh
python tools/serve_device_demo.py --serial YOUR_SERIAL
```

Open **http://localhost:8001/demo_combined.html** and click **기기 앱 연결**.
Choose B or C, train, then run inference; **B/C 학습·실행** runs both in sequence.
Current device defaults are1epoch/batch4/lr2e-4/seed42. The web reads actual device
state and selected settings. Python reference scores in
`docs/experiments/20261016_bc_observed.md` used batch2 and are not device results.

`--port`, `--package`, `--adb` and `--runs-dir` can be set explicitly.
`ANDROID_SERIAL` is accepted instead of `--serial`. Local state/event logs go to
`.runtime/runs` by default. The server derives its web root from its own file,
so it can be launched from any working directory. It listens on loopback only.

## Start the optional screen-mirroring server

The left pane uses ws-scrcpy; model controls and output use the Python server.
The tested ws-scrcpy source is pinned to
`cd6cea6ebde17af6685c2b325183fc24bef9d52b` from NetrisTV/ws-scrcpy. The checked-in
patch reproduces the Android-only build and loopback listener used here.
In a terminal with Node/npm, git, adb and node-gyp build prerequisites:

```sh
# Run from Applications/demo.
git clone https://github.com/NetrisTV/ws-scrcpy.git .tools/ws-scrcpy
git -C .tools/ws-scrcpy checkout cd6cea6ebde17af6685c2b325183fc24bef9d52b
git -C .tools/ws-scrcpy apply ../../tools/ws-scrcpy.patch
cd .tools/ws-scrcpy
npm ci
npm run dist
cd dist
WS_SCRCPY_CONFIG=/absolute/path/to/Applications/demo/tools/ws-scrcpy.config.json node index.js
```

On PowerShell set `$env:WS_SCRCPY_CONFIG` to that absolute config path before
`node index.js`. ws-scrcpy is a separate dependency; its code and `node_modules`
are not vendored into NNTrainer. The bundled patch/config is verified against
the pinned source; rebuilding this dependency on another OS is not a tested
native model backend configuration.

Default mirroring is http://localhost:8002. Override it with
`demo_combined.html?server=http://localhost:18002/` if forwarding a different port.
The device serial comes from `/api/status`, not a hardcoded phone identifier.

For a Linux host with the USB device and a browser on another machine:

```sh
ssh -N -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 USER@HOST
```

Then open the same localhost URL in the browser. Refresh with Ctrl+F5 after
updating files. Do not expose either loopback service directly as a public site.

## Verification and assets

```sh
python tools/verify_bundle.py
node --check device_demo.js
```

The bundle check verifies the source/native data hashes, Android catalog and
local assets. It does not run device training or inference. Actual
Android/model/backend compatibility must be checked on the target device. The original reference test result for B
was Hit@4 97.5%, business Top1 6/8 and personal Top1 8/8 after7epochs; it is not a
guarantee for another device or fewer updates.

The static page assets are local. App icon origin URLs, packages and SHA256 are
in `assets/icons/sources.json`; app names/icons identify their respective apps.
SamsungOne fonts, the supplied phone frame and supplied training diagram are
included as the existing demo's presentation assets. The diagram and frame are
illustrations, not telemetry from the device.
