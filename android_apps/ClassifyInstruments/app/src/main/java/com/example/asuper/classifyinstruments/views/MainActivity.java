package com.example.asuper.classifyinstruments.views;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
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
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.example.asuper.classifyinstruments.models.Classification;
import com.example.asuper.classifyinstruments.models.Classifier;
import com.example.asuper.classifyinstruments.models.TensorFlowClassifier;


public class MainActivity extends Activity implements View.OnClickListener {

	private static final int PIXEL_WIDTH = 28;

	private Button classBtn;
	private TextView classText;
	private Classifier myClassifier;
	public static int MY_PERMISSIONS_REQUEST_CAMERA;
	ImageView imageView;
    Bitmap processed_image = null;



	@Override
	protected void onCreate(Bundle savedInstanceState) {

		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		classBtn = findViewById(R.id.btn_class);
		classBtn.setOnClickListener(this);

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

		if(bmp.getWidth() > bmp.getHeight()){
		    bmp = RotateBitmap(bmp, 90);
        }


		int h = bmp.getHeight();
		int w = bmp.getWidth();

		int center_h = h / 2;


		bmp = Bitmap.createBitmap(bmp, 0,center_h - w/2, w, w);
		bmp = convertGrayScale(bmp);
		bmp = Bitmap.createScaledBitmap(bmp, 28, 28, false);


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

//			float pixels[] = {-0.67f, -0.70f, -0.72f, -0.75f, -0.76f, -0.78f, -0.78f, -0.79f, -0.80f, -0.80f, -0.79f, -0.83f, -0.70f, -0.46f, -0.44f, -0.37f, -0.60f, -0.79f, -0.53f, -0.39f, 0.855f, 0.370f, -0.34f, -0.43f, 1.337f, 1.826f, -0.37f, -0.06f, -0.77f, -0.79f, -0.81f, -0.83f, -0.85f, -0.85f, -0.86f, -0.88f, -0.73f, -0.74f, -0.90f, -0.86f, -0.56f, -0.35f, -0.31f, 0.069f, 0.918f, 0.091f, -0.35f, -0.64f, 1.276f, 2.480f, -0.40f, -0.20f, -0.29f, 2.223f, 0.159f, -0.28f, -0.82f, -0.85f, -0.87f, -0.87f, -0.90f, -0.93f, -0.89f, -0.73f, -0.69f, -0.62f, -0.78f, -0.59f, -0.75f, 0.604f, 2.374f, 2.035f, 2.996f, 1.798f, -0.31f, -0.40f, -0.59f, 2.604f, 0.590f, -0.44f, -0.63f, 0.760f, 1.466f, -0.50f, -0.92f, -0.95f, -0.95f, -0.96f, -0.96f, -0.80f, -0.62f, -0.45f, 0.983f, 1.962f, 0.647f, -0.37f, -0.66f, -0.14f, 2.545f, 2.183f, 2.253f, 2.560f, 0.225f, -0.40f, -0.83f, 0.893f, 2.142f, -0.59f, -0.32f, -0.61f, 1.682f, 0.138f, -0.90f, -0.74f, -0.25f, 0.498f, 0.893f, -0.28f, -0.50f, -0.71f, 1.566f, 2.591f, 1.875f, -0.31f, -0.37f, -0.89f, 1.289f, 2.529f, 1.870f, 2.541f, 1.345f, -0.51f, -0.49f, -0.60f, 2.299f, 0.356f, -0.54f, -0.69f, 0.375f, 1.512f, 1.395f, 1.469f, 1.312f, 2.388f, 2.400f, 0.407f, -0.44f, -0.79f, 0.176f, 2.415f, 2.512f, 0.329f, -0.42f, -0.78f, -0.06f, 2.471f, 1.896f, 2.124f, 2.268f, -0.07f, -0.47f, -0.87f, 0.771f, 1.888f, -0.65f, -0.38f, -0.69f, 1.436f, 2.349f, 2.292f, 1.240f, 1.979f, 2.237f, 1.257f, -0.42f, -0.59f, -0.67f, 1.775f, 2.641f, 1.364f, -0.50f, -0.48f, -0.84f, 1.517f, 2.262f, 1.741f, 2.434f, 1.043f, -0.59f, -0.55f, -0.64f, 2.167f, 0.264f, -0.60f, -0.57f, -0.23f, 1.991f, 2.240f, 1.513f, 1.652f, 2.169f, 1.942f, -0.11f, -0.53f, -0.95f, 0.586f, 2.406f, 2.147f, -0.06f, -0.45f, -0.88f, 0.140f, 2.330f, 1.683f, 2.110f, 2.088f, -0.23f, -0.49f, -0.91f, 0.778f, 1.789f, -0.75f, -0.41f, -0.63f, 1.319f, 2.208f, 1.886f, 1.303f, 2.050f, 2.160f, 0.614f, -0.57f, -0.76f, -0.46f, 1.919f, 2.367f, 0.942f, -0.60f, -0.60f, -0.79f, 1.624f, 2.042f, 1.735f, 2.426f, 0.742f, -0.65f, -0.58f, -0.64f, 2.044f, 0.123f, -0.66f, -0.48f, 0.227f, 2.066f, 2.121f, 1.284f, 1.793f, 2.069f, 1.502f, -0.43f, -0.50f, -0.91f, 0.926f, 2.331f, 1.880f, -0.36f, -0.49f, -0.97f, 0.316f, 2.251f, 1.637f, 2.139f, 1.795f, -0.41f, -0.52f, -0.93f, 0.708f, 1.639f, -0.81f, -0.44f, -0.60f, 1.615f, 2.127f, 1.521f, 1.353f, 1.967f, 2.018f, 0.073f, -0.57f, -0.84f, -0.24f, 2.014f, 2.235f, 0.450f, -0.72f, -0.76f, -0.72f, 1.721f, 1.840f, 1.663f, 2.164f, 0.501f, -0.68f, -0.66f, -0.74f, 1.921f, 0.003f, -0.74f, -0.96f, 0.709f, 2.058f, 1.832f, 1.083f, 1.810f, 2.015f, 0.953f, -0.64f, -0.58f, -0.85f, 1.211f, 2.155f, 1.417f, -0.66f, -0.60f, -1.03f, 0.529f, 2.091f, 1.376f, 1.942f, 1.598f, -0.53f, -0.55f, -1.04f, 0.597f, 1.430f, -0.90f, -0.90f, -0.21f, 1.855f, 1.956f, 1.171f, 1.489f, 1.873f, 1.712f, -0.40f, -0.55f, -0.88f, 0.124f, 1.978f, 1.937f, 0.012f, -0.72f, -0.82f, -0.54f, 1.783f, 1.504f, 1.552f, 2.076f, 0.314f, -0.70f, -0.67f, -0.77f, 1.707f, -0.04f, -0.78f, -0.79f, 1.280f, 1.929f, 1.454f, 1.058f, 1.803f, 2.013f, 0.292f, -0.77f, -0.70f, -0.70f, 1.426f, 1.990f, 0.984f, -0.87f, -0.67f, -1.10f, 0.652f, 1.827f, 1.248f, 1.893f, 1.391f, -0.69f, -0.55f, -1.01f, 0.346f, 1.436f, -0.68f, -0.97f, 0.310f, 1.854f, 1.651f, 0.880f, 1.607f, 1.837f, 1.178f, -0.82f, -0.62f, -0.99f, 0.369f, 1.819f, 1.630f, -0.49f, -0.80f, -0.93f, -0.54f, 1.684f, 1.233f, 1.461f, 1.913f, 0.035f, -0.82f, -0.76f, -0.75f, 1.642f, -0.69f, -0.79f, -0.51f, 1.588f, 1.709f, 1.034f, 1.189f, 1.702f, 1.698f, -0.34f, -0.76f, -0.80f, -0.55f, 1.482f, 1.742f, 0.520f, -1.04f, -0.69f, -1.13f, 0.700f, 1.541f, 1.027f, 1.724f, 1.125f, -0.88f, -0.85f, -0.40f, 1.459f, -0.78f, -0.61f, -0.91f, 0.889f, 1.664f, 1.299f, 0.730f, 1.539f, 1.663f, 0.517f, -1.00f, -0.62f, -0.96f, 0.671f, 1.608f, 1.325f, -0.86f, -0.86f, -1.08f, -0.54f, 1.455f, 0.932f, 1.259f, 1.660f, -0.23f, -1.16f, -0.16f, 1.526f, -1.15f, -0.63f, -0.90f, -0.02f, 1.497f, 1.409f, 0.612f, 1.207f, 1.487f, 1.256f, -0.94f, -0.73f, -0.86f, -0.33f, 1.269f, 1.419f, -0.07f, -1.23f, -1.04f, -1.15f, 0.846f, 1.152f, 0.857f, 1.488f, 0.819f, 0.311f, 1.057f, 1.335f, -1.10f, -0.81f, -0.68f, -0.74f, 1.135f, 1.466f, 0.880f, 0.657f, 1.412f, 1.512f, -0.28f, -1.18f, -0.53f, -0.89f, 0.611f, 1.180f, 0.711f, -1.23f, -1.21f, -1.19f, 0.609f, 1.269f, 0.731f, 1.123f, 1.291f, 1.414f, 1.243f, 1.073f, -0.13f, -1.04f, -0.57f, -0.98f, 0.410f, 1.444f, 1.172f, 0.228f, 1.187f, 1.374f, 0.627f, -1.58f, -0.63f, -1.04f, -0.36f, 0.996f, 0.978f, -0.67f, -1.64f, -0.92f, 0.673f, 1.103f, 0.829f, 0.740f, 1.162f, 1.078f, 1.170f, 0.959f, 0.575f, -1.19f, -0.64f, -0.84f, -0.40f, 1.261f, 1.287f, 0.286f, 0.583f, 1.140f, 1.043f, -1.17f, -1.70f, -1.65f, -0.96f, 0.813f, 0.796f, 0.327f, -0.41f, 0.757f, 0.936f, 0.883f, 0.970f, 0.544f, 0.960f, 1.070f, 1.093f, 1.098f, 0.985f, -0.99f, -0.92f, -0.54f, -0.90f, 0.781f, 1.233f, 0.665f, -0.08f, 0.885f, 0.939f, -0.33f, -2.15f, -2.03f, -1.63f, 0.423f, 0.649f, 0.707f, 0.745f, 0.544f, 0.828f, 0.816f, 0.931f, 0.709f, 0.658f, 1.090f, 1.037f, 1.119f, 1.014f, -0.38f, -1.30f, -0.47f, -0.77f, -0.00f, 1.093f, 0.886f, -0.28f, 0.527f, 0.805f, 0.466f, -1.58f, -0.93f, -0.43f, 0.614f, 0.576f, 0.579f, 0.647f, 0.434f, 0.733f, 0.804f, 0.877f, 0.957f, 0.543f, 1.007f, 1.082f, 1.136f, 0.771f, 0.241f, -1.46f, -0.95f, -1.06f, -0.89f, 0.942f, 0.958f, 0.155f, 0.081f, 0.851f, 0.646f, 0.629f, 0.928f, 0.064f, 0.391f, 0.675f, 0.645f, 0.765f, 0.637f, 0.653f, 0.920f, 0.930f, 1.073f, 0.772f, 0.774f, 1.150f, 1.165f, 0.606f, 0.592f, -1.06f, -1.47f, -1.29f, -1.36f, 0.362f, 0.974f, 0.655f, -0.09f, 0.809f, 0.726f, 0.751f, 0.791f, 0.458f, 0.154f, 0.825f, 0.766f, 0.829f, 0.865f, 0.671f, 0.960f, 0.997f, 1.052f, 1.097f, 0.796f, 1.380f, 1.333f, 0.582f, 0.709f, -0.24f, -1.35f, -1.20f, -1.09f, 0.286f, 0.970f, 0.933f, 0.169f, 0.585f, 0.904f, 0.846f, 0.879f, 0.905f, 0.234f, 0.820f, 0.960f, 0.974f, 1.038f, 0.865f, 0.989f, 1.320f, 1.398f, 1.357f, 0.637f, -0.00f, -0.56f, 0.720f, 0.786f, 0.573f, -0.76f, -0.19f, 0.825f, 0.943f, 0.915f, 1.017f, 0.674f, 0.367f, 1.082f, 1.015f, 1.035f, 1.129f, 0.690f, 0.699f, 1.171f, 1.099f, 1.242f, 1.312f, 1.030f, 0.808f, 0.069f, -0.65f, -1.12f, -1.10f, -1.06f, 0.896f, 0.889f, 0.989f, 0.848f, 0.388f, 1.052f, 0.996f, 1.088f, 1.151f, 1.171f, 0.515f, 1.181f, 1.306f, 1.304f, 1.353f, 1.276f, 0.757f, 1.355f, 1.393f, 1.045f, 0.240f, -0.62f, -1.10f, -1.06f, -1.03f, -0.99f, -0.90f, -0.88f};

            int[] intPix = new int[784];
            processed_image.getPixels(intPix, 0, processed_image.getWidth(), 0, 0, processed_image.getWidth(), processed_image.getHeight());

            float[] pixels = new float[intPix.length];

            for(int i = 0; i < intPix.length; i++) {

                int pix = intPix[i];
                int b = pix & 0xff;
                pixels[i] = (float)((0xff - b) / 255.0);
            }



			String text = "";
			final Classification res = myClassifier.recognize(pixels);

			if (res.getLabel() == null) {
				text += myClassifier.name() + ": Unknown\n";
			} else {
				if(res.getLabel().equals("0")){
					text += "Piano\n";
				}
				else if(res.getLabel().equals("1")){
					text += "Guitar\n";
				}

				text += ("Probability:" + res.getConf()*100 + "%");
			}
			classText.setText(text);
		}
	}
}