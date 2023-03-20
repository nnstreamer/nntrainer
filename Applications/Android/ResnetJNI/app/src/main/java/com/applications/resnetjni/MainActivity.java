package com.applications.resnetjni;

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
        System.loadLibrary("resnet_jni");
    }

    public native long createModel(String in_shape, int unit);

    public native int train_resnet(String[] args, long model_pointer);

    public native String testResnet(String[] args, long model_pointer);

    public native String inferResnet(String[] args, Bitmap bmp, long model_pointer);

    public native String getTrainingStatus(long model_pointer, int cur_iter, int batch_size);

    public native int getCurrentEpoch(long model_pointer);

    public native int requestStop();

    public native String getTestingResult();

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

	EditText edit_batch = (EditText)findViewById(R.id.batch_size);
	EditText edit_split = (EditText)findViewById(R.id.data_split);
	EditText edit_epoch = (EditText)findViewById(R.id.epoch);
	EditText edit_save = (EditText)findViewById(R.id.save);
	EditText edit_save_best = (EditText)findViewById(R.id.save_best);
	EditText edit_w = (EditText)findViewById(R.id.in_w);
	EditText edit_h = (EditText)findViewById(R.id.in_h);
	EditText edit_c = (EditText)findViewById(R.id.in_c);
	EditText edit_data_path = (EditText)findViewById(R.id.save_path);


	Button mkdir_btn =(Button)findViewById(R.id.mkdir_data);
	folder_name = getApplicationContext().getFilesDir().getPath().toString()+"/"+edit_data_path.getText();
	
	train_folder = folder_name + "/train";
	test_folder = folder_name + "/test";

	mkdir_btn.setOnClickListener(new View.OnClickListener(){
		@Override
		public void onClick(View v){
		    File newFolder = new File(folder_name);
		    try{
			newFolder.mkdir();
			Log.d("nntrainer", "Create NNTrainer Data Folder"+folder_name);
			
		    }catch (Exception e){
			Log.d("nntrainer", "Already Exist"+folder_name);
		    }
		    
		    try{
			copyAssetFolder(getApplicationContext(), "train", train_folder);
			copyAssetFolder(getApplicationContext(), "test", test_folder);			

		    }catch (Exception e){
			e.printStackTrace();
		    }
		    
		    Log.d("nntrainer", "Training Data is Ready");

		    String s="";
		    
		    File train_class = new File(train_folder);
		    File[] class_list = train_class.listFiles();
		    num_class = 0;
		    List<Integer>num_data_list  = new ArrayList<Integer>();
		    
		    height = edit_h.getText().toString();
		    width = edit_w.getText().toString();
		    channel=edit_c.getText().toString();

		    String ss ="";
		    for(File c: class_list){
			num_class ++;
			ss+=num_class+ "\t : "+c.getName()+": ";
			File[] f_list = c.listFiles();
			int num_data=0;
			for(File f: f_list){
			    num_data++;
			}
			ss+="( "+num_data+" )\n";
			num_data_list.add(num_data);
		    }
		    
		    s ="Total Number of Class is "+num_class+" \n";
		    s+= "---------------------------------------------------\n";
		    s+=ss;
		    data_view.setText(s);
		}
	    });



        Button train_resnet_btn = (Button)findViewById(R.id.train_resnet);
        train_resnet_btn.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                String program_name = "nntrainer_resnet";
                String batch_size = edit_batch.getText().toString();
                String data_split = edit_split.getText().toString();
                String epoch = edit_epoch.getText().toString();
		String bin_path=getApplicationContext().getFilesDir().getPath().toString()+"/"+edit_save.getText().toString();
		String bin_best_path=getApplicationContext().getFilesDir().getPath().toString()+"/"+edit_save_best.getText().toString();
		String in_shape = channel+":"+height+":"+width;

		if(!training_started && !testing_ing && modelDestroyed()){
		    Training_log = "NNTrainer Start Training\n";
		    Log.d("nntrainer", "create Model");
		    training_started=true;
		    training_finished=false;		    
		    cur_iter=0;
		    model_pointer = createModel(in_shape, num_class);
		    Training_log += "Model Created \n";
		    Log.d("nntrainer", "create Model Done "+model_pointer);
		    
		    String[] args = {program_name, folder_name, batch_size, data_split, epoch, channel, height, width, bin_path, bin_best_path, String.valueOf(num_class)};

		    
		    Thread th = new Thread(new Runnable(){
			    @Override
			    public void run(){
				stop=false;
				training_ing = true;
				Log.d("nntrainer", "running "+model_pointer+" "+num_class);
				
				int status=train_resnet(args, model_pointer);
				
				training_finished=true;
				training_started = false;
				training_ing = false;
			    }
			    
			});

		    th.start();
		
		    Thread th_ui = new Thread(new Runnable(){
			    @Override
			    public void run(){
				
				while (!training_finished){
				    try{
					Thread.sleep(1000);
					int e = getCurrentEpoch(model_pointer);
					
					if(cur_epoch != e){
					    cur_iter = 0;
					    cur_epoch = e ;
					}
					    
					String s="";
					
					if(!modelDestroyed() && stop){
					    s="... Stopping \n";
					} else if(modelDestroyed() && stop){
					    s="Training Stoped\n";
					} else {
					    s=getTrainingStatus(model_pointer,cur_iter, Integer.parseInt(batch_size));
					}
					
					if(!s.equals("-")){
					    Training_log += s;
					    data_view.setText(Training_log);
					    cur_iter++;
					}
				    }catch(Exception e){
					e.printStackTrace();
				    }
				}
			    }
			});
		    th_ui.start();
		}
	    }
	    });


        Button stop_train_btn = (Button)findViewById(R.id.train_stop);
        stop_train_btn.setOnClickListener(new View.OnClickListener() {
		@Override
		public void onClick(View v) {
		    Thread th = new Thread(new Runnable(){
			    public void run(){
				requestStop();
				stop=true;
			    }
			});
		    th.start();
		}
	    });


	Button testing_resnet_btn = (Button)findViewById(R.id.testing_resnet);
        testing_resnet_btn.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
		Log.d("nntrainer", "Training Fished : "+String.valueOf(training_finished));
		testing_done=false;
		stop = false;
                String program_name = "nntrainer_resnet";
                String batch_size = edit_batch.getText().toString();
                String data_split = edit_split.getText().toString();
                String epoch = edit_epoch.getText().toString();
		String bin_path=getApplicationContext().getFilesDir().getPath().toString()+"/"+edit_save.getText().toString();
		String bin_best_path=getApplicationContext().getFilesDir().getPath().toString()+"/"+edit_save_best.getText().toString();
		String in_shape = channel+":"+height+":"+width;

		if(!training_ing && !testing_ing && modelDestroyed()){

		    Testing_log = "NNTrainer Testing\n";
		    model_pointer = createModel(in_shape, num_class);
		    String[] args = {program_name, folder_name, batch_size, data_split, epoch, channel, height, width, bin_path, bin_best_path, String.valueOf(num_class)};
		    
		    Thread th_test = new Thread(new Runnable(){
			    @Override
			    public void run(){
				testing_ing = true;
				Log.d("nnttrainer", String.valueOf(testing_done));				
				Testing_log=testResnet(args, model_pointer);
				testing_done=true;
				testing_ing = false;
			    }
			});

		    th_test.start();
		    Thread th_test_ui = new Thread(new Runnable(){
			    @Override
			    public void run(){
				while(!testing_done){
				    try{
					Thread.sleep(1000);
					if(!stop){
					    Testing_log = getTestingResult();
					    data_view.setText(Testing_log);
					}
				    } catch (Exception e){
					e.printStackTrace();
				    }
				}

			    }
			});
		    th_test_ui.start();
		}
	    }
	    });

	Button infer_btn = (Button)findViewById(R.id.infer_resnet);
	infer_btn.setOnClickListener(new View.OnClickListener(){
		@Override
		public void onClick(View v){
		    Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
		    intent.setType("image/*");
		    intent.setAction(Intent.ACTION_GET_CONTENT);
		    startActivityForResult(intent, 1);
		}
	    });

    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
	super.onActivityResult(requestCode, resultCode, data);
	if(requestCode == 1 && resultCode == RESULT_OK){
	    if(data!= null){
		Log.d("nntrainer", data.getData().toString());
		EditText edit_save_best=(EditText)findViewById(R.id.save_best);
		String bin_best_path=getApplicationContext().getFilesDir().getPath().toString()+"/"+edit_save_best.getText().toString();

		String in_shape = channel+":"+height+":"+width;


		if(!training_ing && !testing_ing && modelDestroyed()){

		    model_pointer = createModel(in_shape, num_class);

		    String in_file = getFileName(data.getData());

		    Infer_log = "NNTrainer Infering: \n"+in_file+"\n\n";

		    Bitmap bmp = BitmapFactory.decodeFile(in_file);
		    Bitmap resizedBmp = Bitmap.createScaledBitmap(bmp, 256,256, true);
		    
		    String[] args = {in_file, channel, height, width,  String.valueOf(num_class), bin_best_path};

		    Infer_log+=inferResnet(args, resizedBmp, model_pointer);
		    
		}
		
		data_view.setText(Infer_log);
	    }
	}
	    
    }

    public String getFileName(Uri uri){

	String result=null;

	if (uri.getScheme() != null && uri.getScheme().equals("content")) {
	    Cursor cursor = getContentResolver().query(uri, null, null, null, null);
	    try {
		if (cursor != null && cursor.moveToFirst()) {
		    //local filesystem
		    int index = cursor.getColumnIndex("_data");
		    if(index == -1)
		    	//google drive
		    	index = cursor.getColumnIndex("_display_name");
		    result = cursor.getString(index);
		    Log.d("nntrainer", result);
		}
	    } finally {
		cursor.close();
	    }
	}

	return result;
    }
    

    

}
