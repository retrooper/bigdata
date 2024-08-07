package com.github.retrooper.bigdata.image;

public class ImageFeatures {
    private final double[] data;

    public ImageFeatures(double[] data) {
        this.data = data;
    }

    public double[] getData() {
        return data;
    }

    public double[] extractData(int limit) {
        double[] array = new double[limit];
        System.arraycopy(getData(), 0, array, 0, limit);
        return array;
    }
}
