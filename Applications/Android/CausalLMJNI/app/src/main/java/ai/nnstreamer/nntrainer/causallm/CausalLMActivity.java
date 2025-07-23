// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.nnstreamer.nntrainer.causallm;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.os.AsyncTask;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.ProgressBar;
import android.widget.CheckBox;
import android.widget.Toast;
import android.util.Log;

public class CausalLMActivity extends AppCompatActivity {
    
    private static final String TAG = "CausalLMActivity";
    
    static {
        System.loadLibrary("causallm_jni");
    }
    
    private EditText inputText;
    private TextView outputText;
    private Button runButton;
    private Button initButton;
    private ProgressBar progressBar;
    private CheckBox sampleCheckBox;
    
    private String modelPath = "/sdcard/nntrainer/causallm/qwen3-4b/";
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_causallm);
        
        initViews();
        setupListeners();
    }
    
    private void initViews() {
        inputText = findViewById(R.id.input_text);
        outputText = findViewById(R.id.output_text);
        runButton = findViewById(R.id.run_button);
        initButton = findViewById(R.id.init_button);
        progressBar = findViewById(R.id.progress_bar);
        sampleCheckBox = findViewById(R.id.sample_checkbox);
        
        // Set default input
        inputText.setText("<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n");
    }
    
    private void setupListeners() {
        initButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new InitModelTask().execute();
            }
        });
        
        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isModelInitialized()) {
                    String input = inputText.getText().toString();
                    boolean doSample = sampleCheckBox.isChecked();
                    new RunInferenceTask().execute(input, String.valueOf(doSample));
                } else {
                    Toast.makeText(CausalLMActivity.this, "Model not initialized", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
    
    private class InitModelTask extends AsyncTask<Void, Void, Boolean> {
        @Override
        protected void onPreExecute() {
            progressBar.setVisibility(View.VISIBLE);
            initButton.setEnabled(false);
            outputText.setText("Initializing model...");
        }
        
        @Override
        protected Boolean doInBackground(Void... voids) {
            return initializeModel(modelPath);
        }
        
        @Override
        protected void onPostExecute(Boolean success) {
            progressBar.setVisibility(View.GONE);
            initButton.setEnabled(true);
            
            if (success) {
                outputText.setText("Model initialized successfully!");
                runButton.setEnabled(true);
                Toast.makeText(CausalLMActivity.this, "Model ready for inference", Toast.LENGTH_SHORT).show();
            } else {
                outputText.setText("Failed to initialize model. Please check model files.");
                runButton.setEnabled(false);
                Toast.makeText(CausalLMActivity.this, "Initialization failed", Toast.LENGTH_SHORT).show();
            }
        }
    }
    
    private class RunInferenceTask extends AsyncTask<String, Void, String> {
        @Override
        protected void onPreExecute() {
            progressBar.setVisibility(View.VISIBLE);
            runButton.setEnabled(false);
            outputText.setText("Running inference...");
        }
        
        @Override
        protected String doInBackground(String... params) {
            String input = params[0];
            boolean doSample = Boolean.parseBoolean(params[1]);
            return runInference(input, doSample);
        }
        
        @Override
        protected void onPostExecute(String result) {
            progressBar.setVisibility(View.GONE);
            runButton.setEnabled(true);
            outputText.setText(result);
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        destroyModel();
    }
    
    // Native methods
    public native boolean initializeModel(String modelPath);
    public native String runInference(String inputText, boolean doSample);
    public native void destroyModel();
    public native boolean isModelInitialized();
}