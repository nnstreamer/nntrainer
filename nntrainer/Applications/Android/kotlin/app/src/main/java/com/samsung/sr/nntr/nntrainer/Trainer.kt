package com.samsung.sr.nntr.nntrainer

import android.content.Context
import android.util.Log

class NNTrainerNative {
   init{
      System.loadLibrary("nntrainer_resnet")
   }

   external fun nntrainerTrain(tdata: FloatArray, batch_size:Int, data_size: Int, data_len: Int, label_len: Int, dir_path: String, file_path: String, best_path:String, epoch: Int ): Int

   external fun nntrainerTest(vdata: FloatArray, output: FloatArray, data_size: Int, data_len: Int, label_len: Int, dir_path:String, file_path: String, correct:Int ): Int
}


class Trainer(private val context: Context ) {

    fun train() {
    	val batch_size = 16
	val epoch = 2
	var model_bin_path = context.filesDir.toString()+"/weight.bin"
	var dir_path = context.filesDir.toString()
	var model_bin_best_path = context.filesDir.toString()+"/resnet_best.bin"
	var d_size:Int = 512
	var image_size : Int =  32*32*3
	var label_len:Int = 100
	Log.i("nntrainer","d_size : " + d_size.toString()+" "+label_len)
	var dummyArray :FloatArray = FloatArray(1)
	NNTrainerNative().nntrainerTrain(dummyArray, batch_size, d_size , image_size, label_len, dir_path, model_bin_path, model_bin_best_path, epoch)

    }

    fun testing():String{
	var dir_path = context.filesDir.toString()
	var model_bin_path = context.filesDir.toString()+"/resnet_best.bin"
	var d_size : Int =10
	var l_len : Int =100
	var correct:Int = 0
	var image_size : Int = 32*32*3
	Log.i("nntrainer","d_size : " + d_size.toString()+" "+l_len)
        var outputArray = FloatArray(d_size * l_len)
	var dummyArray : FloatArray = FloatArray(1)
	var result = NNTrainerNative().nntrainerTest(dummyArray, outputArray, d_size, image_size, l_len, dir_path, model_bin_path, correct)
	var ss =""
	var right:Float = result.toFloat()
	var accuracy = right /d_size *100.0

	ss = "Accuracy : " +"%.3f".format(accuracy) +"% "+"%d".format(result)+"\n" + ss
	return ss
    }
}
