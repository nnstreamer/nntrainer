package com.samsung.sr.nntr.nntrainer

import android.graphics.Bitmap
import android.content.Context
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import com.samsung.sr.nntr.nntrainer.Trainer
import kotlinx.coroutines.*
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.*

@Composable
fun TrainingScreen(trainer: Trainer) {
    val padding = 16.dp
    val trainDisabled = remember { mutableStateOf(false) }
    val testingDisabled = remember { mutableStateOf(false) }
    var (log_s, setLog_s) = remember{mutableStateOf<String>("")}

    fun train() {
        trainer.train()	
        trainDisabled.value = false
	setLog_s("Training is Done")
    }

    fun testing(){
        var ss = trainer.testing()
        testingDisabled.value = false
	setLog_s(ss)
    }

    Column(
        modifier = Modifier.padding(horizontal = padding)
    ) {
        Row {
            Button(
                onClick = {
                    trainDisabled.value = true
                    CoroutineScope(newSingleThreadContext("train")).launch { train() } },
                modifier = Modifier.padding(4.dp),
                enabled = !trainDisabled.value
            ) {
                Text("TRAINING")
            }
	    
            Button(
                onClick = {
                    testingDisabled.value = true
                    CoroutineScope(newSingleThreadContext("testing")).launch { testing() } },
                modifier = Modifier.padding(4.dp),
                enabled = !testingDisabled.value
            ) {
                Text("TESTING")
            }
        }
	Text(log_s)
    }
}
