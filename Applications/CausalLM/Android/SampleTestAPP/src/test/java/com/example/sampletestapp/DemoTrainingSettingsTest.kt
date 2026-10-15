// SPDX-License-Identifier: Apache-2.0
package com.example.sampletestapp

import org.junit.Assert.*
import org.junit.Test

class DemoTrainingSettingsTest {
    private fun defaults(user: String) = mapOf<String, Any>(
        "data_path" to "/data/$user/train.jsonl", "valid_data" to "/data/$user/validation.jsonl",
        "validate" to true, "output_path" to "/adapters/$user.bin", "model_path" to "/model/cpu",
        "hybrid_trunk" to "/model/gpu", "task" to "decision", "objective" to "soft",
        "epochs" to 1, "batch_size" to 4, "lr" to 2e-4, "seq_len" to 208,
        "lora_rank" to 8, "lora_alpha" to 16, "lora_last_n" to 4,
        "lora_qat" to true, "lora_weight_q4" to true, "seed" to 42,
    )

    private fun settings() = DemoTrainingSettings(mapOf("u02" to defaults("u02"), "u03" to defaults("u03")))

    @Test fun combinedRunFromBRetainsEditedSettingsWhenItReachesC() {
        val settings = settings()
        val edited = defaults("u02") + mapOf(
            "epochs" to 3, "batch_size" to 8, "lr" to 1.5e-4, "seq_len" to 240,
            "lora_rank" to 16, "lora_alpha" to 32, "lora_last_n" to 2,
            "lora_qat" to false, "lora_weight_q4" to false, "patience" to 2,
            "clip_grad" to 0.5, "max_samples" to 100, "train_norms" to true,
            "lora_path" to "/resume/B.bin",
        )
        val run = settings.snapshot("u02", edited, combined = true)
        for (user in listOf("u02", "u03")) {
            for (key in listOf("epochs", "batch_size", "lr", "seq_len", "lora_rank", "lora_alpha",
                "lora_last_n", "lora_qat", "lora_weight_q4", "patience", "clip_grad", "max_samples", "train_norms")) {
                assertEquals("$user/$key", edited[key], run.getValue(user)[key])
                assertEquals("UI $user/$key", edited[key], settings.load(user)[key])
            }
            assertEquals("/data/$user/train.jsonl", run.getValue(user)["data_path"])
            assertEquals("/data/$user/validation.jsonl", run.getValue(user)["valid_data"])
            assertEquals("/adapters/$user.bin", run.getValue(user)["output_path"])
        }
        assertEquals("/resume/B.bin", run.getValue("u02")["lora_path"])
        assertFalse(run.getValue("u03").containsKey("lora_path"))
    }

    @Test fun combinedRunFromCUsesCSettingsForBothAndKeepsSeparateResumeAdapters() {
        val settings = settings()
        settings.save("u02", defaults("u02") + ("lora_path" to "/resume/B.bin"))
        val edited = defaults("u03") + mapOf("epochs" to 5, "batch_size" to 2, "lr" to 9e-5, "lora_path" to "/resume/C.bin")
        val run = settings.snapshot("u03", edited, combined = true)
        for (user in listOf("u02", "u03")) {
            assertEquals(5, run.getValue(user)["epochs"])
            assertEquals(2, run.getValue(user)["batch_size"])
            assertEquals(9e-5, run.getValue(user)["lr"])
            assertEquals("/data/$user/train.jsonl", run.getValue(user)["data_path"])
        }
        assertEquals("/resume/B.bin", run.getValue("u02")["lora_path"])
        assertEquals("/resume/C.bin", run.getValue("u03")["lora_path"])
    }

    @Test fun individualCommandsKeepIndependentUserDrafts() {
        val settings = settings()
        settings.save("u03", defaults("u03") + ("epochs" to 7))
        val run = settings.snapshot("u02", defaults("u02") + ("epochs" to 3), combined = false)
        assertEquals(3, run.getValue("u02")["epochs"])
        assertEquals(7, run.getValue("u03")["epochs"])
        assertEquals(7, settings.load("u03")["epochs"])
    }

    @Test fun acceptedRunIsIsolatedFromUiSwitchesAndSubsequentEdits() {
        val settings = settings()
        val edited = (defaults("u02") + ("epochs" to 3)).toMutableMap()
        val run = settings.snapshot("u02", edited, combined = true)
        // The UI saves the departing user, then loads C's draft on a phase switch.
        settings.save("u02", edited)
        assertEquals(3, settings.load("u03")["epochs"])
        edited["epochs"] = 9
        settings.save("u03", defaults("u03") + ("epochs" to 8))
        assertEquals(3, run.getValue("u02")["epochs"])
        assertEquals(3, run.getValue("u03")["epochs"])
        assertEquals(3, settings.load("u02")["epochs"])
        assertEquals(8, settings.load("u03")["epochs"])
    }

    @Test fun seedIsFixedAndNonDecisionSettingsDoNotReplaceDecisionDrafts() {
        val settings = settings()
        settings.save("u03", defaults("u03") + mapOf("epochs" to 2, "seed" to 999))
        val run = settings.snapshot("u03", null, combined = true)
        for (user in listOf("u02", "u03")) {
            assertEquals(2, run.getValue(user)["epochs"])
            assertEquals("decision", run.getValue(user)["task"])
            assertEquals(42, run.getValue(user)["seed"])
            assertEquals(42, settings.load(user)["seed"])
        }
    }
}
