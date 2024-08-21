package com.github.retrooper.bigdata.util;

public class ArrayConversions {
    public static float[][] convertTwoDTwoF(double[][] doubleArray) {
        int rows = doubleArray.length;
        int cols = doubleArray[0].length;
        float[][] floatArray = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                floatArray[i][j] = (float) doubleArray[i][j];
            }
        }

        return floatArray;
    }

    public static double[][] convertTwoFTwoD(float[][] floatArray) {
        int rows = floatArray.length;
        int cols = floatArray[0].length;
        double[][] doubleArray = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                doubleArray[i][j] = (double) floatArray[i][j];
            }
        }

        return doubleArray;
    }

    public static float[] convertDToF(double[] data) {
        int length = data.length;
        float[] floatArray = new float[length];

        for (int i = 0; i < length; i++) {
            floatArray[i] = (float) data[i];
        }

        return floatArray;
    }


    public static double[] convertFToD(float[] data) {
        int length = data.length;
        double[] doubleArray = new double[length];

        for (int i = 0; i < length; i++) {
            doubleArray[i] = (double) data[i];
        }

        return doubleArray;
    }
}
