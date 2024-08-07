package com.github.retrooper.bigdata.algorithm.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset2D;

import java.util.Map;

public class LinearRegressionAlgorithm implements LearningAlgorithm<Float> {
    private final float r;
    private final float gradient;
    private final float height;

    private LinearRegressionAlgorithm(float r, float gradient, float height) {
        this.r = r;
        this.gradient = gradient;
        this.height = height;
    }

    public static LinearRegressionAlgorithm build(FunctionDataset2D function) {
        float xSum = 0, ySum = 0, xSqSum = 0, ySqSum = 0, xySum = 0;
        for (Map.Entry<Float, Float> entry : function.getData().entrySet()) {
            float x = entry.getKey();
            float y = entry.getValue();
            xSum += x;
            ySum += y;
            xSqSum += x * x;
            ySqSum += y * y;
            xySum += x * y;
        }

        int n = function.dataPoints();

        float xMean = xSum / n;
        float yMean = ySum / n;

        float Sxx = xSqSum - n * xMean * xMean;
        float Syy = ySqSum - n * yMean * yMean;
        float Sxy = xySum - n * xMean * yMean;
        // Correlation coefficient
        float r = (float) (Sxy / Math.sqrt(Sxx * Syy));

        // Gradient of this function
        float gradient = Sxy / Sxx;

        float height = yMean - gradient * xMean;
        return new LinearRegressionAlgorithm(r, gradient, height);
    }

    @Override
    public float predict(Float x) {
        return gradient() * x + height();
    }

    public float r() {
        return r;
    }

    public float gradient() {
        return gradient;
    }

    public float height() {
        return height;
    }
}
