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

import java.io.BufferedReader;
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

			List<Float> mean = null, std = null;
			try {
                mean = getPopMean(getAssets(), "popMean.txt");
                std = getPopStd(getAssets(), "popSTD.txt");
                pixels = normalizeImage(pixels, mean, std);
            } catch(Exception e){
			    e.printStackTrace();
            }


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

        float[] pixels = new float[intPix.length];

        for(int i = 0; i < intPix.length; i++) {

            int pix = intPix[i];
            int b = pix & 0xff;
            pixels[i] = (float)((0xff - b) / 255.0);
        }

        return pixels;
    }

    private static List<Float> getPopMean(AssetManager a, String file) throws IOException{

	    List<Float> mean = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(a.open(file)));

        String line = reader.readLine();
        while(line != null) {

            line = line.replace("[", "");
            line = line.replace("]", "");
            String[] nums = line.split("\\s+");

            for(String s : nums){

                if(s!=null && !s.equals("")){
                    mean.add(Float.parseFloat(s));
                }

            }
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

            line = line.replace("[", "");
            line = line.replace("]", "");
            String[] nums = line.split("\\s+");

            for(String s : nums){

                if(s!=null && !s.equals("")){
                    std.add(Float.parseFloat(s));
                }

            }
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
