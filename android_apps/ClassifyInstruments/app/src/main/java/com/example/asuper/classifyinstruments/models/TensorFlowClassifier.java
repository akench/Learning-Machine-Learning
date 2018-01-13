package com.example.asuper.classifyinstruments.models;

import android.content.res.AssetManager;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;


import android.content.res.AssetManager;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Changed from https://github.com/MindorksOpenSource/AndroidTensorFlowMNISTExample/blob/master
 * /app/src/main/java/com/mindorks/tensorflowexample/TensorFlowImageClassifier.java
 * Created by marianne-linhares on 20/04/17.
 */

public class TensorFlowClassifier implements Classifier {

    private static final float THRESHOLD = 0.5f;

    private TensorFlowInferenceInterface tfHelper;

    private String name;
    private String inputName;
    private String outputName;
    private int inputSize;

    private List<String> labels;
    private float[] output;
    private String[] outputNames;

    private static List<String> readLabels(AssetManager am, String fileName) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(am.open(fileName)));

        String line;
        List<String> labels = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            labels.add(line);
            Log.d("label", "LABEL!!!!!!  " + line);
        }

        br.close();
        return labels;
    }

    public static TensorFlowClassifier create(AssetManager assetManager, String name,
                                              String modelPath, String labelFile,
                                              int inputSize, String inputName, String outputName)
                                                throws IOException {
        TensorFlowClassifier classifier = new TensorFlowClassifier();

        classifier.name = name;

        classifier.inputName = inputName;
        classifier.outputName = outputName;

        classifier.labels = readLabels(assetManager, labelFile);

        classifier.tfHelper = new TensorFlowInferenceInterface(assetManager, modelPath);
        int numClasses = 2;

        classifier.inputSize = inputSize;

        classifier.outputNames = new String[] { outputName };

        classifier.outputName = outputName;
        classifier.output = new float[numClasses];


        return classifier;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public Classification recognize(final float[] pixels) {

        tfHelper.feed(inputName, pixels, 1, inputSize, inputSize, 1);

        tfHelper.feed("keep_prob", new float[] { 1 });

        tfHelper.run(outputNames);

        tfHelper.fetch(outputName, output);


        Classification ans = new Classification();

        for (int i = 0; i < output.length; ++i) {
            System.out.println(output[i]);
            System.out.println(labels.get(i));
            if (output[i] > THRESHOLD && output[i] > ans.getConf()) {
                ans.update(output[i], labels.get(i));
            }
        }

        return ans;
    }
}
