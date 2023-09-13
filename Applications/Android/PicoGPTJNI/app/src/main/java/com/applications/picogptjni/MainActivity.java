package com.applications.picogptjni;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.content.res.AssetManager;
import android.content.res.AssetFileDescriptor;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.FileUtils;
import android.os.Handler;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.NumberPicker;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipEntry;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("picogpt_jni");
    }

    public native long createModel();

    public native String inferPicoGPT(String path, String sentence, long model_pointer);

    public native String getResult(long model_pointer);

    public native String getInferResult();    

    public native boolean modelDestroyed();

    TextView data_view;
    int view_id;

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case 0:
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED) {

                } else {
                }
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + requestCode);
        }
    }


    public static boolean copyAssetFolder(Context context, String srcName, String dstName) {
	try {
	    boolean result = true;
	    String fileList[] = context.getAssets().list(srcName);
	    if (fileList == null) return false;

	    if (fileList.length == 0) {
		result = copyAssetFile(context, srcName, dstName);
	    } else {
		File file = new File(dstName);
		result = file.mkdirs();
		for (String filename : fileList) {
		    result &= copyAssetFolder(context, srcName + File.separator + filename, dstName + File.separator + filename);
		}
	    }
	    return result;
	} catch (IOException e) {
	    e.printStackTrace();
	    return false;
	}
    }

    public static boolean copyAssetFile(Context context, String srcName, String dstName) {
	try {
	    InputStream in = context.getAssets().open(srcName);
	    File outFile = new File(dstName);
	    OutputStream out = new FileOutputStream(outFile);
	    byte[] buffer = new byte[1024];
	    int read;
	    while ((read = in.read(buffer)) != -1) {
		out.write(buffer, 0, read);
	    }
	    in.close();
	    out.close();
	    return true;
	} catch (IOException e) {
	    e.printStackTrace();
	    return false;
	}
    }
    


    String folder_name;
    String train_folder;
    String test_folder;
    long model_pointer;

    String Training_log;
    String Testing_log;
    String Infer_log;        
    boolean training_finished=false;
    boolean training_started=false;
    boolean training_ing=false;
    boolean testing_ing=false;    
    boolean testing_done=false;    
    int cur_iter = 0;
    int cur_epoch = 1;
    int num_class;
    String height;
    String width;
    String channel;
    boolean stop=false;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
	
	data_view = (TextView) findViewById(R.id.data_out);
	
	data_view.setMovementMethod(new ScrollingMovementMethod());

	FrameLayout frame = (FrameLayout) findViewById(R.id.frame);
	
	view_id = 0;

        String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE};
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, permissions, 0);
        }

	EditText in_text = (EditText)findViewById(R.id.input_text);

	folder_name = getApplicationContext().getFilesDir().getPath().toString();

	if(! (new File(folder_name+"/pico_gpt.bin").exists()))
	    copyAssetFolder(getApplicationContext(), "pico_gpt.bin", folder_name+"/pico_gpt.bin");

	if(! (new File(folder_name+"/merges.txt").exists()))
	    copyAssetFolder(getApplicationContext(), "merges.txt", folder_name+"/merges.txt");

	if(! (new File(folder_name+"/vocab.json").exists()))	
	    copyAssetFolder(getApplicationContext(), "vocab.json", folder_name+"/vocab.json");	

	    
	Button infer_btn = (Button)findViewById(R.id.infer_picogpt);
	
	infer_btn.setOnClickListener(new View.OnClickListener(){
		@Override
		public void onClick(View v){
		    
		    model_pointer = createModel();
		    
		    Thread th_run = new Thread(new Runnable(){
			    @Override
			    public void run(){
				training_finished=false;
				EditText in_text = (EditText)findViewById(R.id.input_text);		
				String path=getApplicationContext().getFilesDir().getPath().toString();

				Infer_log = "NNTrainer Infering: \n\n";

				Infer_log+=inferPicoGPT(path, in_text.getText().toString(), model_pointer);
				training_finished=true;
			    }
			});

		    th_run.start();

		    Thread th_ui = new Thread(new Runnable(){
			    @Override
			    public void run(){
				while(!training_finished){
				    try{Thread.sleep(100);
					String s=getInferResult();
					data_view.setText(s);
				    } catch(Exception e){
					e.printStackTrace();
				    }
				}
			    }
			});
		    th_ui.start();
		}
	    });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
	super.onActivityResult(requestCode, resultCode, data);
	if(requestCode == 1 && resultCode == RESULT_OK){
	    if(data!= null){
		Log.d("nntrainer", data.getData().toString());
		EditText in_text = (EditText)findViewById(R.id.input_text);		
		String path=getApplicationContext().getFilesDir().getPath().toString();


		if(!training_ing && !testing_ing && modelDestroyed()){

		    model_pointer = createModel();

		    Infer_log = "NNTrainer Infering: \n\n";

		    Infer_log+=inferPicoGPT(path, in_text.getText().toString(), model_pointer);
		    
		}
		
		data_view.setText(Infer_log);
	    }
	}
	    
    }


}
