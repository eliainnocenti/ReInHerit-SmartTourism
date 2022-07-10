package org.tensorflow.lite.examples.classification.tflite;

import java.util.ArrayList;

public class Element {
    private String style;
    private  String color;
    private double distance;
    private ArrayList<Float> matrix;


    public Element (String style, String color,ArrayList<Float> matrix, double distance){
        this.color = color;
        this.style = style;
        this.distance = distance;
        this.matrix = matrix;
    }

    public double getDistance() {
        return distance;
    }

    public String getColor() {
        return color;
    }

    public String getStyle() {
        return style;
    }

    public ArrayList<Float> getMatrix(){ return matrix;}

    @Override
    public String toString() {
        return "Element{" +
                "style='" + style + '\'' +
                ", color='" + color + '\'' +
                ", distance=" + distance +
                '}';
    }
}
