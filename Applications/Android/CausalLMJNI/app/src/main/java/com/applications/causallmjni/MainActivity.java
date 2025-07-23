package com.applications.causallmjni;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "CausalLMJNI";
    
    // Load native library
    static {
        System.loadLibrary("causallm_jni");
    }

    // Native method declarations
    public native long createCausalLMModel(String configPath);
    public native int initializeModel(long modelPointer);
    public native int loadWeights(long modelPointer, String weightPath);
    public native String runInference(long modelPointer, String inputText, boolean doSample);
    public native void destroyModel(long modelPointer);

    private long modelPointer = 0;
    private EditText editTextInput;
    private TextView textViewOutput;
    private Button buttonInitialize;
    private Button buttonRun;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI components
        editTextInput = findViewById(R.id.editTextInput);
        textViewOutput = findViewById(R.id.textViewOutput);
        buttonInitialize = findViewById(R.id.buttonInitialize);
        buttonRun = findViewById(R.id.buttonRun);

        // Set up button listeners
        buttonInitialize.setOnClickListener(v -> initializeModelAsync());
        buttonRun.setOnClickListener(v -> runInferenceAsync());

        Log.d(TAG, "MainActivity created");
    }

    private void initializeModelAsync() {
        new Thread(() -> {
            try {
                // TODO: Set the correct path to your model configuration
                String configPath = "/sdcard/causallm_models/qwen3-4b";
                String weightPath = configPath + "/nntr_qwen3_4b_fp32.bin";

                runOnUiThread(() -> {
                    buttonInitialize.setEnabled(false);
                    buttonInitialize.setText("Initializing...");
                });

                Log.d(TAG, "Creating CausalLM model from: " + configPath);
                modelPointer = createCausalLMModel(configPath);
                
                if (modelPointer == 0) {
                    throw new RuntimeException("Failed to create CausalLM model");
                }

                Log.d(TAG, "Initializing model...");
                int initResult = initializeModel(modelPointer);
                if (initResult != 0) {
                    throw new RuntimeException("Failed to initialize model");
                }

                Log.d(TAG, "Loading weights from: " + weightPath);
                int loadResult = loadWeights(modelPointer, weightPath);
                if (loadResult != 0) {
                    throw new RuntimeException("Failed to load weights");
                }

                runOnUiThread(() -> {
                    buttonInitialize.setText("Model Initialized");
                    buttonRun.setEnabled(true);
                    Toast.makeText(this, "Model initialized successfully!", Toast.LENGTH_SHORT).show();
                });

                Log.d(TAG, "Model initialization completed successfully");

            } catch (Exception e) {
                Log.e(TAG, "Error during model initialization", e);
                runOnUiThread(() -> {
                    buttonInitialize.setEnabled(true);
                    buttonInitialize.setText("Initialize Model");
                    Toast.makeText(this, "Initialization failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }

    private void runInferenceAsync() {
        String inputText = editTextInput.getText().toString().trim();
        if (inputText.isEmpty()) {
            Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show();
            return;
        }

        new Thread(() -> {
            try {
                runOnUiThread(() -> {
                    buttonRun.setEnabled(false);
                    buttonRun.setText("Running...");
                    textViewOutput.setText("Generating response...");
                });

                Log.d(TAG, "Running inference with input: " + inputText);
                String result = runInference(modelPointer, inputText, false); // Use greedy decoding

                runOnUiThread(() -> {
                    textViewOutput.setText(result);
                    buttonRun.setEnabled(true);
                    buttonRun.setText("Run Inference");
                });

                Log.d(TAG, "Inference completed");

            } catch (Exception e) {
                Log.e(TAG, "Error during inference", e);
                runOnUiThread(() -> {
                    textViewOutput.setText("Error: " + e.getMessage());
                    buttonRun.setEnabled(true);
                    buttonRun.setText("Run Inference");
                });
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (modelPointer != 0) {
            destroyModel(modelPointer);
            modelPointer = 0;
            Log.d(TAG, "Model destroyed");
        }
    }
}