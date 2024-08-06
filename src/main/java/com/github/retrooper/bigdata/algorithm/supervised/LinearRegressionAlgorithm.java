package com.github.retrooper.bigdata.algorithm.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset;

import java.util.Map;

public class LinearRegressionAlgorithm implements LearningAlgorithm {
    private final double r;
    private final double gradient;
    private final double height;

    private LinearRegressionAlgorithm(double r, double gradient, double height) {
        this.r = r;
        this.gradient = gradient;
        this.height = height;
    }

    public static LinearRegressionAlgorithm build(FunctionDataset<Double, Double> function) {
        double xSum = 0, ySum = 0, xSqSum = 0, ySqSum = 0, xySum = 0;
        for (Map.Entry<Double, Double> entry : function.getData().entrySet()) {
            double x = entry.getKey();
            double y = entry.getValue();
            xSum += x;
            ySum += y;
            xSqSum += x * x;
            ySqSum += y * y;
            xySum += x * y;
        }

        int n = function.dataPoints();

        double xMean = xSum / n;
        double yMean = ySum / n;

        double Sxx = xSqSum - n * xMean * xMean;
        double Syy = ySqSum - n * yMean * yMean;
        double Sxy = xySum - n * xMean * yMean;
        // Correlation coefficient
        double r = Sxy / Math.sqrt(Sxx * Syy);

        // Gradient of this function
        double gradient = Sxy / Sxx;

        double height = yMean - gradient * xMean;
        return new LinearRegressionAlgorithm(r, gradient, height);
    }

    @Override
    public double predict(double x) {
        return gradient() * x + height();
    }

    public double r() {
        return r;
    }

    public double gradient() {
        return gradient;
    }

    public double height() {
        return height;
    }
}
