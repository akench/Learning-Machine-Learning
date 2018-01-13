package com.example.asuper.classifyinstruments.views;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Bundle;

import com.example.asuper.classifyinstruments.R;

/*
   Copyright 2016 Narrative Nights Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   From: https://raw.githubusercontent
   .com/miyosuda/TensorFlowAndroidMNIST/master/app/src/main/java/jp/narr/tensorflowmnist
   /DrawModel.java
*/

import android.app.Activity;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import com.example.asuper.classifyinstruments.models.Classification;
import com.example.asuper.classifyinstruments.models.Classifier;
import com.example.asuper.classifyinstruments.models.TensorFlowClassifier;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends Activity implements View.OnClickListener {

	private static final int PIXEL_WIDTH = 28;

	private Button classBtn;
	private ImageButton resetBtn;
	private TextView classText;
	private Classifier myClassifier, tempClassifier;
	public static int MY_PERMISSIONS_REQUEST_CAMERA;
	ImageView imageView;
	Bitmap processed_image = null;



	@Override
	protected void onCreate(Bundle savedInstanceState) {


		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		classBtn = findViewById(R.id.btn_class);
		classBtn.setOnClickListener(this);

		resetBtn = findViewById(R.id.reset_btn);
		resetBtn.setOnClickListener(this);

		classText = findViewById(R.id.tfRes);
		imageView = findViewById(R.id.image_view);


		loadModel();


		while (ContextCompat.checkSelfPermission(this,
				Manifest.permission.CAMERA)
				!= PackageManager.PERMISSION_GRANTED) {

			ActivityCompat.requestPermissions(this,
					new String[]{Manifest.permission.CAMERA},
					MY_PERMISSIONS_REQUEST_CAMERA);

			try{
				Thread.sleep(1000);
			}catch(Exception e){}

		}

		Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
		startActivityForResult(intent, 0);

	}


	protected void onActivityResult(int requestCode, int resultCode, Intent data) {

		Bitmap bmp = (Bitmap) data.getExtras().get("data");

        int h = bmp.getHeight();
        int w = bmp.getWidth();

        int center_h = h / 2;
        int center_w = w / 2;

		if(bmp.getWidth() > bmp.getHeight()){
			bmp = Bitmap.createBitmap(bmp, center_w - h/2, 0, h, h);
			bmp = convertGrayScale(bmp);
			bmp = Bitmap.createScaledBitmap(bmp, PIXEL_WIDTH, PIXEL_WIDTH, false);
		}
        else {
            bmp = Bitmap.createBitmap(bmp, 0,center_h - w/2, w, w);
            bmp = convertGrayScale(bmp);
            bmp = Bitmap.createScaledBitmap(bmp, PIXEL_WIDTH, PIXEL_WIDTH, false);

        }

		imageView.setImageBitmap(bmp);
		processed_image = bmp;
	}


	private Bitmap convertGrayScale(Bitmap src){

		float[] matrix = new float[]{
				0.3f, 0.59f, 0.11f, 0, 0,
				0.3f, 0.59f, 0.11f, 0, 0,
				0.3f, 0.59f, 0.11f, 0, 0,
				0, 0, 0, 1, 0,};

		Bitmap gray = Bitmap.createBitmap(
				src.getWidth(),
				src.getHeight(),
				src.getConfig());

		Canvas canvas = new Canvas(gray);
		Paint paint = new Paint();
		ColorMatrixColorFilter filter = new ColorMatrixColorFilter(matrix);
		paint.setColorFilter(filter);
		canvas.drawBitmap(src, 0, 0, paint);

		return gray;
	}

	public static Bitmap RotateBitmap(Bitmap source, float angle){
		Matrix matrix = new Matrix();
		matrix.postRotate(angle);
		return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
	}




	@Override
	protected void onResume() {
		super.onResume();
	}

	@Override
	protected void onPause() {
		super.onPause();
	}

	private void loadModel() {

		new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					myClassifier = TensorFlowClassifier.create(getAssets(), "TensorFlow",
							"opt_pvg_model.pb", "labels.txt", PIXEL_WIDTH,
							"input", "output");
				} catch (final Exception e) {
					throw new RuntimeException("Error initializing classifiers!", e);
				}
			}
		}).start();
	}

	@Override
	public void onClick(View view) {

		if (view.getId() == R.id.btn_class) {


			float[] pixels = bitmapToPixels(processed_image);

			for(float f : pixels){
			    Log.d("value", "" + f);
            }

			List<Float> mean = null, std = null;
			try {
                mean = getPopMean(getAssets(), "popMean.txt");
                std = getPopStd(getAssets(), "popSTD.txt");
                pixels = normalizeImage(pixels, mean, std);
            } catch(Exception e){
			    Log.d("line", "excepton!!!");
				e.printStackTrace();
            }

            for(float f : pixels) {
                Log.d("Float", f + "");
            }


//            float[] p = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,6,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,9,10,10,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,17,52,132,189,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7,0,0,0,47,129,180,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,2,10,20,23,26,57,39,9,0,0,8,0,0,0,0,0,0,0,0,0,0,0,2,7,9,5,6,0,1,17,121,183,121,65,68,56,35,40,0,0,0,0,0,0,0,0,0,0,0,1,9,20,19,14,17,45,5,0,33,115,188,153,67,52,65,62,4,0,0,0,0,0,0,1,0,0,1,5,12,27,76,148,132,113,119,21,0,0,69,170,179,98,37,58,17,0,0,0,0,0,3,5,2,0,9,47,99,143,209,220,222,130,109,140,46,2,0,55,128,178,147,62,25,0,0,0,0,2,6,14,4,0,16,101,164,191,189,209,208,219,109,106,95,0,11,0,41,170,217,198,171,2,5,1,1,12,44,104,58,0,3,1,57,163,178,194,212,210,215,99,73,14,0,0,117,220,205,210,217,9,13,1,6,57,131,207,144,4,0,2,4,97,177,171,200,206,210,212,93,28,84,138,219,213,210,210,208,44,79,4,10,40,47,168,178,50,27,8,9,39,128,173,170,213,212,219,210,193,229,219,192,213,215,210,210,160,183,5,0,22,11,124,201,89,31,30,28,8,82,199,189,185,222,212,216,219,210,213,212,197,211,215,214,140,177,20,5,19,6,84,189,136,36,32,9,7,171,221,221,179,201,221,214,214,215,213,213,212,202,216,202,120,192,73,30,30,15,60,156,190,28,15,85,170,226,216,218,214,177,217,216,215,214,218,225,218,164,103,137,113,205,98,31,51,7,45,189,222,153,119,233,227,216,218,216,221,202,187,223,221,225,201,152,101,87,72,123,106,200,131,19,27,13,158,232,215,240,168,187,223,217,218,216,215,225,197,205,193,127,91,91,112,133,111,106,101,208,144,27,85,195,226,219,219,222,201,160,226,215,218,226,226,205,134,97,134,107,127,137,136,133,129,89,176,221,198,202,193,226,219,221,220,219,222,158,211,228,219,181,129,95,101,84,137,146,137,135,134,133,139,109,218,214,219,224,188,213,222,220,219,219,235,187,176,177,109,96,110,131,145,113,117,150,134,135,139,147,142,111,211,212,212,218,186,205,220,224,232,222,186,109,82,140,129,143,145,141,139,133,98,153,143,148,135,100,48,12,210,212,212,215,189,203,228,207,160,113,95,119,88,137,151,142,141,141,138,148,109,138,128,74,29,0,0,0,208,210,214,227,190,176,156,101,104,127,144,153,121,112,155,139,142,149,152,140,91,41,16,0,0,0,1,1,216,224,212,170,97,84,142,141,151,149,146,145,142,99,155,155,150,125,77,30,3,0,0,1,1,0,0,0,192,147,104,96,125,101,138,156,147,147,147,147,157,120,124,109,51,13,0,0,0,1,1,0,0,0,0,0,97,105,133,149,155,131,112,159,147,149,150,151,134,82,25,4,0,0,1,1,0,0,0,0,0,0,0,0};
//            pixels = normalizeImage(p, mean, std);

            String text = "";
			final Classification res = myClassifier.recognize(pixels);

			if (res.getLabel() == null) {
				text += myClassifier.name() + ": Error\n";
			} else {
				if(res.getLabel().equals("0")){
					text += "Piano\n";
				}
				else if(res.getLabel().equals("1")){
					text += "Guitar\n";
				}

				text += ("Probability:" + res.getConf()*100 + "%\n");
			}
			classText.setText(text);


		}

		else if(view.getId() == R.id.reset_btn){

			Intent i = getBaseContext().getPackageManager()
					.getLaunchIntentForPackage(getBaseContext().getPackageName() );
			i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
			startActivity(i);
		}

	}


	private static float[] bitmapToPixels(Bitmap bmp){

        int[] intPix = new int[PIXEL_WIDTH * PIXEL_WIDTH];
        bmp.getPixels(intPix, 0, bmp.getWidth(), 0, 0,
                bmp.getWidth(), bmp.getHeight());


        for(int i : intPix){
            Log.d("intpix", ""+i);
        }

        float[] pixels = new float[intPix.length];

        for(int i = 0; i < intPix.length; i++) {

            int pix = intPix[i];
            int b = pix & 0xff;
            pixels[i] = (float)(b);
        }

        return pixels;
    }

    private static List<Float> getPopMean(AssetManager a, String file) throws IOException{

	    List<Float> mean = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(a.open(file)));

        String line = reader.readLine();
        while(line != null) {

            mean.add(Float.parseFloat(line));
            Log.d("hihihi", "" + line);
            line = reader.readLine();
        }
        reader.close();
        return mean;
    }

    private static List<Float> getPopStd(AssetManager a, String file) throws IOException{

        List<Float> std = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(a.open(file)));

        String line = reader.readLine();
        while(line != null) {

            std.add(Float.parseFloat(line));
            Log.d("byebye", "" + line);
            line = reader.readLine();
        }
        reader.close();
        return std;
    }

    private static float[] normalizeImage(float[] pixels, List<Float> mean, List<Float> std){

	    for(int i = 0; i < pixels.length; i++){
	        pixels[i] -= mean.get(i);
	        pixels[i] /= std.get(i);
        }

        return pixels;

    }
}
