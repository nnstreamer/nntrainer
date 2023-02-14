package com.applications.resnetjni;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.NumberPicker;
import android.widget.TextView;
import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("resnet_jni");
    }

    public native int train_resnet(String[] args);

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case 0:
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                    // write here

                } else {
                    // show message to user that permission needed
                }
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + requestCode);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE};
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, permissions, 0);
        }

        NumberPicker batch_size_np = (NumberPicker) findViewById(R.id.batch_size);
        batch_size_np.setMinValue(1);
        batch_size_np.setMaxValue(512);
        batch_size_np.setValue(1);

        NumberPicker data_split_np = (NumberPicker) findViewById(R.id.data_split);
        data_split_np.setMinValue(1);
        data_split_np.setMaxValue(100);
        data_split_np.setValue(1);

        NumberPicker epoch_np = (NumberPicker) findViewById(R.id.epoch);
        epoch_np.setMinValue(1);
        epoch_np.setMaxValue(10000);
        epoch_np.setValue(1);

        Button train_resnet_btn = (Button)findViewById(R.id.train_resnet);
        train_resnet_btn.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {

                String program_name = "nntrainer_resnet";
                String dataset_path = "fake";
                String batch_size = Integer.toString(batch_size_np.getValue());
                String data_split = Integer.toString(data_split_np.getValue());
                String epoch = Integer.toString(epoch_np.getValue());

                String[] args = {program_name, dataset_path, batch_size, data_split, epoch};
                train_resnet(args);
            }
        });


    }
}