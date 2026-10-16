// SPDX-License-Identifier: Apache-2.0
package com.example.sampletestapp

/** Shuffle seed of every demo training run (fixed so runs are reproducible). */
internal const val DEMO_SEED = 1

/** Editable per-user drafts and immutable command snapshots of flat training parameters. */
internal class DemoTrainingSettings(initial: Map<String, Map<String, Any>>) {
    private val drafts = initial.mapValues { fixedSeed(it.value) }.toMutableMap()

    fun save(userId: String, settings: Map<String, Any>) {
        require(userId in drafts) { "Unknown demo user: $userId" }
        drafts[userId] = fixedSeed(settings)
    }

    fun load(userId: String): Map<String, Any> = drafts.getValue(userId).toMap()

    /** A combined run shares the visible Settings, but never another user's data or adapter. */
    fun snapshot(
        currentUser: String,
        current: Map<String, Any>?,
        combined: Boolean,
    ): Map<String, Map<String, Any>> {
        if (current != null) save(currentUser, current)
        if (combined) {
            val common = drafts.getValue(currentUser)
            for (userId in drafts.keys) {
                val own = drafts.getValue(userId)
                val merged = common.toMutableMap()
                for (key in USER_FIELDS) {
                    if (key in own) merged[key] = own.getValue(key) else merged.remove(key)
                }
                drafts[userId] = fixedSeed(merged)
            }
        }
        // UI switches and later edits must not change an already accepted command.
        return drafts.mapValues { it.value.toMap() }
    }

    private fun fixedSeed(settings: Map<String, Any>) = settings.toMutableMap().apply { put("seed", DEMO_SEED) }.toMap()

    private companion object {
        val USER_FIELDS = setOf("data_path", "valid_data", "validate", "output_path", "lora_path")
    }
}
