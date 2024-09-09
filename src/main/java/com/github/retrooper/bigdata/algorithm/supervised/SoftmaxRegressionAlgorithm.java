package com.github.retrooper.bigdata.algorithm.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.util.Arrays;

public class SoftmaxRegressionAlgorithm<Z extends NDimensionalPoint> implements LearningAlgorithm<Z> {

    private float[][] weights;
    private int numClasses;
    private float learningRate;
    private int epochs;


    public SoftmaxRegressionAlgorithm(int numClasses, float learningRate, int epochs) {
        this.numClasses = numClasses;
        this.learningRate = learningRate;
        this.epochs = epochs;

    }

    private float sigmoid(float z) {
        return (float) (1 / (1 + Math.exp(-z)));
    }
    private float[][] softmax(float[][] logits) {
        int numSamples = logits.length;
        int numClasses = logits[0].length;
        float[][] probabilities = new float[numSamples][numClasses];

        for (int i = 0; i < numSamples; i++) {
            float[] row = logits[i];
            float max = row[0];
            //Find max
            for (float f : row) {
                if (f > max) {
                    max = f;
                }
            }
            float sum = 0;

            // Compute the softmax scores
            for (int j = 0; j < numClasses; j++) {
                probabilities[i][j] = (float) Math.exp(row[j] - max);
                sum += probabilities[i][j];
            }
            // Normalize the scores
            for (int j = 0; j < numClasses; j++) {
                probabilities[i][j] /= sum;
            }
        }

        return probabilities;
    }

    // Cross-entropy loss function
    private double crossEntropyLoss(float[][] yTrue, float[][] yPred) {
        int numSamples = yTrue.length;
        double loss = 0;

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numClasses; j++) {
                loss -= yTrue[i][j] * Math.log(yPred[i][j]);
            }
        }

        return loss / numSamples;
    }

    // Fit the model
    public void fit(float[][] X, int[] y) {
        int numSamples = X.length;
        int numFeatures = X[0].length;

        // Initialize weights randomly
        weights = new float[numFeatures][numClasses];
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numClasses; j++) {
                weights[i][j] = (float) Math.random();
            }
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Compute the logits
            float[][] logits = new float[numSamples][numClasses];
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numClasses; j++) {
                    logits[i][j] = dotProduct(X[i], getColumn(weights, j));
                }
            }

            // Apply softmax to get probabilities
            float[][] probabilities = softmax(logits);

            // Create one-hot encoded true labels
            float[][] yTrue = new float[numSamples][numClasses];
            for (int i = 0; i < numSamples; i++) {
                yTrue[i][y[i]] = 1;
            }

            // Compute gradient and update weights
            float[][] gradients = new float[numFeatures][numClasses];
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numClasses; j++) {
                    double error = probabilities[i][j] - yTrue[i][j];
                    for (int k = 0; k < numFeatures; k++) {
                        gradients[k][j] += error * X[i][k];
                    }
                }
            }
            // Update weights
            for (int i = 0; i < numFeatures; i++) {
                for (int j = 0; j < numClasses; j++) {
                    weights[i][j] -= learningRate * gradients[i][j] / numSamples;
                }
            }

            // Compute and print the loss every 100 epochs
            if (epoch % 100 == 0) {
                double loss = crossEntropyLoss(yTrue, probabilities);
                System.out.println("Epoch " + epoch + ": Loss = " + loss);
            }
        }
    }

    // Predict the class of a new sample
    public float predict(Z x) {
        float[] logits = new float[numClasses];
        for (int i = 0; i < numClasses; i++) {
            logits[i] = dotProduct(x.coordinates(), getColumn(weights, i));
        }
        float[] probabilities = softmax(new float[][] { logits })[0];
        int predictedClass = 0;
        for (int i = 1; i < numClasses; i++) {
            if (probabilities[i] > probabilities[predictedClass]) {
                predictedClass = i;
            }
        }
        return predictedClass;
    }

    // Helper method to compute the dot product of two vectors
    private float dotProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // Helper method to get a column from the weight matrix
    private float[] getColumn(float[][] matrix, int columnIndex) {
        int numRows = matrix.length;
        float[] column = new float[numRows];
        for (int i = 0; i < numRows; i++) {
            column[i] = matrix[i][columnIndex];
        }
        return column;
    }
}
