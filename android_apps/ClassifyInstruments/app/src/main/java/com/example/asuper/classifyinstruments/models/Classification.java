package com.example.asuper.classifyinstruments.models;

/**
 * Created by marianne-linhares on 20/04/17.
 */

public class Classification {

    private float conf;
    private String label;
    private float[] output;

    Classification() {
        this.conf = -1.0F;
        this.label = null;
    }

    public void setOutput(float[] o){
        this.output = o;
    }

    public float[] getOutput(){
        return this.output;
    }

    void update(float conf, String label) {
        this.conf = conf;
        this.label = label;
    }

    public String getLabel() {
        return label;
    }

    public float getConf() {
        return conf;
    }
}
