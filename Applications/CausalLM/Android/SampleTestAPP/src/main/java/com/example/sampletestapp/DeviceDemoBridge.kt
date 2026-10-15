// SPDX-License-Identifier: Apache-2.0
package com.example.sampletestapp

import android.content.Context
import android.content.Intent
import android.os.Debug
import android.os.Build
import android.os.Process
import android.os.SystemClock
import android.util.Base64
import com.example.quickdotai.NativeCausalLm
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.Executor

/** Structured, local ADB transport for the existing native decision/training APIs.
 * No network permission, prompt rewriting, or reference predictions are involved. */
class DeviceDemoBridge(
    private val context: Context,
    private val executor: Executor,
    private val users: List<User>,
    private val trainingSettings: (String) -> Map<String, Map<String, Any>>,
    private val appBusy: () -> Boolean,
    private val changed: (String) -> Unit,
) {
    data class Scenario(val title: String, val situation: String, val answer: Int,
                        val prompt: String, val required: List<Int>, val metadata: JSONObject = JSONObject()) {
        val chip: String get() = title
        val reqTop4: List<Int> get() = required
    }
    data class User(val label: String, val data: String, val scenarios: List<Scenario>,
                    val metadata: JSONObject = JSONObject(), val training: JSONObject = JSONObject()) {
        val trainData: String get() = data
        val id: String get() = "u0${label[0] - 'A' + 1}"
    }
    private val lock = Any()
    private val directory = File(context.getExternalFilesDir(null), "device_demo").apply { mkdirs() }
    private val session = UUID.randomUUID().toString()
    private val runDirectory = File(directory, session).apply { mkdirs() }
    private var sequence = 0
    @Volatile var busy = false; private set
    @Volatile private var stop = false
    // Captured on the UI thread when a command is accepted. Training and
    // inference never read mutable Settings fields from the engine thread.
    private var commandConfigs = emptyList<JSONObject>()
    private val state = JSONObject().put("schema", 1).put("session_id", session)
        .put("controls_version", 3).put("run_user_mode", "inference_only")
        .put("run_all_mode", "train_and_infer_each_user").put("training_seed", 42)
        .put("selected_user", users.first().id)
        .put("source", "android_native").put("pid", Process.myPid())
        .put("dataset_id", ObservedDemoCatalog.DATASET_ID).put("input_schema", ObservedDemoCatalog.INPUT_SCHEMA)
        .put("app_catalog", ObservedDemoCatalog.appCatalog).put("app_catalog_sha256", ObservedDemoCatalog.CATALOG_SHA256)
        .put("dataset_manifest_sha256", ObservedDemoCatalog.DATA_SHA256)
        .put("device_model", Build.MODEL).put("android_version", Build.VERSION.RELEASE)
        .put("package", context.packageName).put("phase", "idle").put("busy", false)
        .put("message", "기기 연결 준비 완료").put("users", JSONArray(users.mapIndexed { index, user ->
            JSONObject().put("id", user.id).put("label", user.label)
                .put("data_path", user.data).put("data_sha256", digest(File(user.data)))
                .put("expected_data_sha256", user.metadata.getString("data_sha256"))
                .put("expected_data_rows", user.metadata.getInt("expected_data_rows"))
                .put("frequency_baseline", user.metadata.getJSONObject("frequency_baseline"))
                .put("persona", user.metadata.getString("persona")).put("train_config", JSONObject(user.training.toString()))
                .put("validation_path", user.training.getString("valid_data"))
                .put("validation_sha256", digest(File(user.training.getString("valid_data"))))
                .put("validation_rows", user.metadata.getInt("validation_rows"))
                .put("test_path", user.metadata.getString("test_path"))
                .put("test_sha256", digest(File(user.metadata.getString("test_path"))))
                .put("test_rows", user.metadata.getInt("test_rows"))
                .put("data_rows", File(user.data).takeIf { it.exists() }?.useLines { it.count { line -> line.isNotBlank() } } ?: 0)
                .put("status", "ready").put("epochs", JSONArray()).put("iterations", JSONArray())
                .put("scenarios", JSONArray(user.scenarios.mapIndexed { s, sc ->
                    JSONObject().put("id", sc.metadata.getString("id")).put("title", sc.title)
                        .put("sample_id", sc.metadata.getString("sample_id")).put("event_id", sc.metadata.getString("event_id"))
                        .put("observed_input", sc.metadata.getJSONObject("observed_input"))
                        .put("case", sc.metadata.getString("case")).put("variant", sc.metadata.getString("variant"))
                        .put("split", sc.metadata.getString("split")).put("featured", sc.metadata.getBoolean("featured"))
                        .put("situation", sc.situation).put("prompt", sc.prompt)
                        .put("prompt_sha256", digest(sc.prompt.toByteArray()))
                        .put("answer_index", sc.answer).put("required_indices", JSONArray(sc.required))
                }))
        }))

    init { users.indices.forEach(::restoreAdapter); publish() }
    /** Trained adapters live next to the dataset so they survive app restarts
     *  and reinstalls; only a successful training run overwrites them. */
    private fun adapterDirectory(u: Int) = File(File(users[u].data).parentFile!!.parentFile, "adapters/${users[u].id}")
    private fun restoreAdapter(u: Int) {
        val meta = File(adapterDirectory(u), "adapter.json")
        val saved = try { JSONObject(meta.readText()) } catch (_: Exception) { null }
        val adapter = saved?.let { File(adapterDirectory(u), it.optString("adapter_file")) }
        synchronized(lock) {
            if (saved == null || adapter == null || !adapter.isFile || digest(adapter) != saved.optString("adapter_sha256")) {
                user(u).put("lora_enabled", false); return
            }
            user(u).put("adapter_path", adapter.path).put("adapter_sha256", saved.getString("adapter_sha256"))
                .put("adapter_saved_at_ms", saved.getLong("saved_at_ms"))
                .put("train_config", saved.getJSONObject("train_config"))
                .put("epochs", saved.optJSONArray("epochs") ?: JSONArray())
                .put("iterations", saved.optJSONArray("iterations") ?: JSONArray())
                .put("status", "trained").put("lora_enabled", true)
        }
    }
    private fun persistAdapter(u: Int) {
        val dir = adapterDirectory(u).apply { mkdirs() }
        val trained = synchronized(lock) { File(user(u).getString("adapter_path")) }
        // Keep the fp32 master next to the q4 adapter the app loads.
        val masters = trained.path.removeSuffix("_q4.bin").let { if (it == trained.path) emptyList() else listOf(File("$it.bin")) }
        val copies = (listOf(trained) + masters.filter { it.isFile }).map { source ->
            val name = if (source == trained) "lora_adapter${if (trained.name.endsWith("_q4.bin")) "_q4" else ""}.bin" else "lora_adapter.bin"
            val temporary = File(dir, "$name.tmp")
            source.copyTo(temporary, overwrite = true)
            temporary to File(dir, name)
        }
        for ((temporary, target) in copies) check(temporary.renameTo(target)) { "Cannot save adapter: ${target.path}" }
        val adapter = copies.first().second
        val sha = digest(adapter)!!
        val now = System.currentTimeMillis()
        val meta = synchronized(lock) {
            user(u).put("adapter_path", adapter.path).put("adapter_sha256", sha).put("adapter_saved_at_ms", now)
            JSONObject().put("adapter_file", adapter.name).put("adapter_sha256", sha).put("saved_at_ms", now)
                .put("train_config", user(u).getJSONObject("train_config"))
                .put("epochs", user(u).getJSONArray("epochs")).put("iterations", user(u).getJSONArray("iterations"))
        }
        val temporary = File(dir, "adapter.json.tmp")
        temporary.writeText(meta.toString())
        check(temporary.renameTo(File(dir, "adapter.json"))) { "Cannot save adapter metadata" }
    }
    private fun digest(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    private fun digest(file: File): String? = if (file.isFile) digest(file.readBytes()) else null
    private fun user(index: Int) = state.getJSONArray("users").getJSONObject(index)
    private fun scenario(u: Int, s: Int) = user(u).getJSONArray("scenarios").getJSONObject(s)
    private fun publish(message: String? = null, phase: String? = null, current: Int? = null) = synchronized(lock) {
        if (message != null) state.put("message", message)
        if (phase != null) state.put("phase", phase)
        if (current != null) state.put("active_user", users[current].id)
        state.put("busy", busy).put("sequence", ++sequence).put("updated_at_ms", System.currentTimeMillis())
            .put("process_pss_kb", Debug.getPss())
        val text = state.toString()
        val temporary = File(directory, "state.tmp")
        temporary.writeText(text)
        check(temporary.renameTo(File(directory, "state.json"))) { "Cannot publish device state" }
        File(runDirectory, "state.json").writeText(text)
        changed(text)
    }
    fun accept(intent: Intent) {
        val raw = intent.getStringExtra("quickai_demo_command_b64") ?: return
        intent.removeExtra("quickai_demo_command_b64")
        try {
            require(raw.length <= 4096) { "Command too large" }
            command(JSONObject(String(Base64.decode(raw, Base64.NO_WRAP), Charsets.UTF_8)))
        } catch (error: Exception) { publish("요청 거부: ${error.message}") }
    }
    fun stopRun() {
        stop = true
        NativeCausalLm.stopLoraTrainingNative()
        publish("정지 요청: 현재 단계 종료 후 멈춥니다")
    }
    fun command(request: JSONObject) {
        val action = request.getString("action")
        require(action in setOf("status", "select", "before", "train", "after", "run_user", "run_all", "lora_on", "lora_off", "stop")) { "Unknown command" }
        val index = users.indexOfFirst { it.id == request.optString("user", "u03") }
        require(index in users.indices) { "Unknown user" }
        val requestId = request.getString("request_id")
        require(requestId.matches(Regex("[A-Za-z0-9_-]{1,80}"))) { "Invalid request ID" }
        if (action == "stop") { stopRun(); return }
        if (action == "status" || action == "select") {
            synchronized(lock) { state.put("ack_request_id", requestId).put("selected_user", users[index].id) }
            publish(); return
        }
        if (busy || appBusy()) {
            synchronized(lock) { state.put("rejected_request_id", requestId) }
            publish("실행 중입니다. 완료 후 다시 요청하세요."); return
        }
        if (action == "lora_on" || action == "lora_off") {
            val enable = action == "lora_on"
            val label = users[index].label
            synchronized(lock) {
                if (enable && !user(index).has("adapter_path")) {
                    state.put("rejected_request_id", requestId)
                    null
                } else {
                    user(index).put("lora_enabled", enable)
                    state.put("ack_request_id", requestId).put("selected_user", users[index].id)
                }
            } ?: run { publish("사용자 $label: 저장된 LoRA가 없습니다 · 먼저 학습하세요"); return }
            publish(if (enable) "사용자 $label LoRA 로드 · 추론 실행 시 저장된 LoRA 적용"
                else "사용자 $label weights 초기화 · 추론 실행 시 기본 모델(LoRA 미적용)")
            return
        }
        if (state.optString("request_id") == requestId) { publish(); return }
        val settings = trainingSettings(action)
        commandConfigs = users.map { definition -> JSONObject(settings.getValue(definition.id)).put("seed", 42) }
        busy = true; stop = false
        synchronized(lock) {
            state.put("request_id", requestId).put("ack_request_id", requestId).put("command", action)
                .put("selected_user", users[index].id)
                .put("command_status", "running").remove("error")
        }
        publish("실행 준비 중", "preparing", index)
        executor.execute {
            try {
                when (action) {
                    "before" -> beforeAll()
                    "train" -> train(index)
                    "after" -> after(index)
                    "run_user" -> if (loraEnabled(index)) after(index) else base(index)
                    "run_all" -> { beforeAll(); for (u in users.indices) { checkStop(); train(u); after(u) } }
                }
                checkStop()
                synchronized(lock) { state.put("command_status", "complete") }
                val message = when (action) {
                    "train" -> "사용자 ${users[index].label} 학습 완료 · LoRA 저장됨 · 추론 실행으로 결과를 확인하세요"
                    "run_user" -> if (loraEnabled(index)) "사용자 ${users[index].label} 추론 완료 · 저장된 LoRA 적용"
                        else "사용자 ${users[index].label} 추론 완료 · 기본 모델(LoRA 미적용)"
                    "run_all" -> "B/C 학습·추론 완료 · 각 사용자 탭에서 결과를 확인하세요"
                    else -> "추론 완료 · 추가 학습 없음"
                }
                busy = false; publish(message, "complete", index)
            } catch (error: Exception) {
                synchronized(lock) { state.put("error", error.message).put("command_status", if (stop) "stopped" else "error") }
                busy = false; publish(if (stop) "실행 중지됨" else "실패: ${error.message}", if (stop) "stopped" else "error", index)
            }
        }
    }
    private fun checkStop() = check(!stop) { "Stopped" }
    private fun parameters(u: Int, output: String): JSONObject {
        val definition = users[u]
        val params = JSONObject(commandConfigs[u].toString())
        check(digest(File(params.getString("data_path"))) == definition.metadata.getString("data_sha256")) { "Training dataset hash mismatch" }
        check(digest(File(params.getString("valid_data"))) == definition.metadata.getString("valid_sha256")) { "Validation dataset hash mismatch" }
        require(params.getInt("epochs") > 0 && params.getInt("batch_size") > 0) { "Epochs and batch size must be positive" }
        params.put("output_path", output).put("seed", 42)
        return params
    }
    private fun load(params: JSONObject, adapter: String?): Long {
        NativeCausalLm.ensureLoaded()
        val handle = NativeCausalLm.loadModelHandleFromDirNative(0, params.getString("model_path"), 1,
            context.applicationInfo.nativeLibraryDir, adapter, params.getInt("lora_rank"),
            params.getInt("lora_alpha"), params.getInt("lora_last_n"))
        check(handle != 0L) { "Model load failed" }
        return handle
    }
    private fun read(handle: Long, u: Int, s: Int, phase: String) {
        checkStop()
        val sc = users[u].scenarios[s]
        val started = SystemClock.elapsedRealtimeNanos()
        val probabilities = NativeCausalLm.decideModelHandleNative(handle, sc.prompt, 14)
            ?: error("Decision failed: ${users[u].label}/$s")
        val milliseconds = (SystemClock.elapsedRealtimeNanos() - started) / 1e6
        require(probabilities.size == 14 && probabilities.all { it.isFinite() && it >= 0 && it <= 1 } &&
            kotlin.math.abs(probabilities.sum() - 1f) < 0.002f) { "Invalid native probabilities: ${users[u].label}/$s" }
        val top = (0..11).sortedWith(compareBy({ -probabilities[it] }, { it })).take(4)
        val result = JSONObject().put("probs", JSONArray(probabilities.map { it.toDouble() }))
            .put("top4_indices", JSONArray(top)).put("elapsed_ms", milliseconds)
            .put("timestamp_ms", System.currentTimeMillis()).put("request_id", state.getString("request_id"))
            .put("prompt_sha256", digest(sc.prompt.toByteArray())).put("source", "android_native")
        synchronized(lock) { scenario(u, s).put(phase, result) }
        publish("${users[u].label} · ${phase.uppercase()} ${s + 1}/${users[u].scenarios.size}", phase, u)
    }
    private fun loraEnabled(u: Int) = synchronized(lock) { user(u).optBoolean("lora_enabled") && user(u).has("adapter_path") }
    private fun beforeAll() {
        val complete = synchronized(lock) { users.indices.all { u -> users[u].scenarios.indices.all { s -> scenario(u,s).has("before") } } }
        if (complete) { publish("이 세션에서 측정한 학습 전 결과를 유지합니다", "before"); return }
        publish("공통 학습 전 추론 준비 · B/C 학습은 수행하지 않습니다", "before")
        val handle = load(parameters(0, File(runDirectory,"unused.bin").path), null)
        try { for (u in users.indices) for (s in users[u].scenarios.indices) read(handle,u,s,"before") }
        finally { NativeCausalLm.destroyModelHandleNative(handle) }
    }
    /** Base model (no LoRA) inference for one user; results fill BEFORE. */
    private fun base(u: Int) {
        checkStop()
        publish("사용자 ${users[u].label} 기본 모델 로드 중 · LoRA 미적용", "before", u)
        val handle = load(parameters(u, File(runDirectory,"unused.bin").path), null)
        try { for (s in users[u].scenarios.indices) read(handle,u,s,"before") }
        finally { NativeCausalLm.destroyModelHandleNative(handle) }
    }
    private fun train(u: Int) {
        try { trainAndSave(u) } catch (error: Exception) {
            // A failed or stopped run keeps the previously saved adapter.
            synchronized(lock) { user(u).put("status", "ready") }
            restoreAdapter(u); throw error
        }
    }
    private fun trainAndSave(u: Int) {
        checkStop()
        val output = File(runDirectory,"adapter_${users[u].label}_${System.currentTimeMillis()}.bin").path
        val params = parameters(u, output)
        require(File(users[u].data).isFile) { "Training data missing" }
        synchronized(lock) {
            user(u).put("epochs", JSONArray()).put("iterations", JSONArray()).put("train_config", params)
                .put("status", "training").remove("adapter_path")
            for (key in listOf("adapter_sha256", "training_wall_ms", "train_return_code", "train_error"))
                user(u).remove(key)
            for (s in users[u].scenarios.indices) scenario(u,s).remove("after")
        }
        publish("사용자 ${users[u].label} 학습 중", "training", u)
        val started = SystemClock.elapsedRealtimeNanos()
        val result = NativeCausalLm.trainLoraNative(params.toString(), object : NativeCausalLm.TrainListener {
            override fun onIterationProgress(iteration: Int, epoch: Int, loss: Float) {
                if (!loss.isFinite()) {
                    synchronized(lock) { user(u).put("train_error", "Non-finite loss") }
                    NativeCausalLm.stopLoraTrainingNative()
                    publish("학습 오류: Non-finite loss", "training", u)
                    return
                }
                synchronized(lock) { user(u).getJSONArray("iterations").put(JSONObject().put("iteration",iteration).put("epoch",epoch).put("loss",loss.toDouble()).put("elapsed_ms",(SystemClock.elapsedRealtimeNanos()-started)/1e6)) }
                publish("${users[u].label} · epoch $epoch · iteration $iteration · loss %.4f".format(loss), "training", u)
            }
            override fun onEpochProgress(epoch: Int, trainLoss: Float, validLoss: Float, saved: Boolean, bestLoss: Float, numSamples: Int, outputPath: String, q4OutputPath: String, epochSeconds: Double, trainSeconds: Double) {
                if (!trainLoss.isFinite() || !epochSeconds.isFinite() || !trainSeconds.isFinite()) {
                    synchronized(lock) { user(u).put("train_error", "Non-finite epoch metric") }
                    NativeCausalLm.stopLoraTrainingNative()
                    publish("학습 오류: Non-finite epoch metric", "training", u)
                    return
                }
                val metric = JSONObject().put("epoch",epoch).put("train_loss",trainLoss.toDouble())
                    .put("valid_loss",if(validLoss.isFinite()) validLoss.toDouble() else JSONObject.NULL)
                    .put("best_loss",if(bestLoss.isFinite()) bestLoss.toDouble() else JSONObject.NULL)
                    .put("saved",saved).put("num_samples",numSamples).put("epoch_seconds",epochSeconds)
                    .put("train_seconds",trainSeconds).put("output_path",outputPath).put("q4_output_path",q4OutputPath)
                synchronized(lock) { user(u).getJSONArray("epochs").put(metric); if(saved) user(u).put("adapter_path", if(q4OutputPath.isNotBlank() && File(q4OutputPath).isFile) q4OutputPath else outputPath) }
                publish("${users[u].label} · epoch $epoch 완료", "training",u)
            }
            override fun onTrainComplete() {}
            override fun onTrainError(message: String) { synchronized(lock) { user(u).put("train_error",message) } }
        })
        synchronized(lock) { user(u).put("training_wall_ms",(SystemClock.elapsedRealtimeNanos()-started)/1e6).put("train_return_code",result) }
        check(result == 0) { "Training failed (rc=$result)" }; checkStop()
        synchronized(lock) {
            check(!user(u).has("train_error")) { user(u).optString("train_error") }
            check(user(u).has("adapter_path") && File(user(u).getString("adapter_path")).isFile) { "Training did not save an adapter" }
            user(u).put("status","trained")
        }
        persistAdapter(u)
        synchronized(lock) { user(u).put("lora_enabled", true) }
        publish("사용자 ${users[u].label} 학습 완료 · LoRA 저장됨", "trained",u)
    }
    private fun after(u: Int) {
        checkStop()
        val adapter = synchronized(lock) { user(u).optString("adapter_path") }
        require(adapter.isNotBlank() && File(adapter).isFile) { "먼저 선택 사용자를 학습하세요" }
        publish("사용자 ${users[u].label} 저장된 LoRA 로드 중", "after",u)
        // Use the adapter's actual training architecture even if the next
        // run's Settings have since been edited.
        val handle = load(JSONObject(user(u).getJSONObject("train_config").toString()),adapter)
        try { for(s in users[u].scenarios.indices) read(handle,u,s,"after") }
        finally { NativeCausalLm.destroyModelHandleNative(handle) }
        synchronized(lock) { user(u).put("status","complete") }
        publish("사용자 ${users[u].label} 전후 비교 준비 완료", "after",u)
    }
}
