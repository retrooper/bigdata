package com.github.retrooper.bigdata.algorithm.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.util.NDimensionalPoint;

public class LogisticRegressionAlgorithm<Z extends NDimensionalPoint> implements LearningAlgorithm<Z> {

    private float[] weights;
    private float bias;
    private float learningRate;

    private int features;


    public LogisticRegressionAlgorithm(int features, float learningRate) {
        this.features = features;
        this.weights = new float[features];
        this.bias = 0.0F;
        this.learningRate = learningRate;

        // Initialize weights randomly
        for (int i = 0; i < features; i++) {
            weights[i] = (float) (Math.random() - 0.5F);
        }

    }

    private float sigmoid(float z) {
        return (float) (1 / (1 + Math.exp(-z)));
    }

    // Train on a single example (Stochastic Gradient Descent)
    public void train(float[] features, int label) {
        // Compute weighted sum of features + bias
        float z = bias;
        for (int i = 0; i < features.length; i++) {
            z += weights[i] * features[i];
        }

        // Prediction
        float prediction = sigmoid(z);

        // Calculate the error (label - prediction)
        float error = label - prediction;

        // Update weights and bias using gradient descent
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error * features[i];
        }
        bias += learningRate * error;
    }


    @Override
    public float predict(Z x) {
        float z = bias;
        for (int i = 0; i < x.coordinates().length; i++) {
            z += weights[i] * x.coordinates()[i];
        }

        return sigmoid(z) >= 0.5 ? 1 : 0;  // Binary classification
    }
}
