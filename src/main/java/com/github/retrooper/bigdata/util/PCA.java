package com.github.retrooper.bigdata.util;

public class PCA {
    public float[][] data;
    public float[] mean;
    public float[][] covarianceMatrix;
    public float[] eigenvalues;
    public float[][] eigenvectors;

    public PCA() {
    }

    public void init() {
        mean = calculateMean();
        calculateCovarianceMatrix();
        calculateEigen();
    }

    // Step 1: Calculate the mean of each feature
    private float[] calculateMean() {
        int numSamples = data.length;
        int numFeatures = data[0].length;
        float[] mean = new float[numFeatures];

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                mean[j] += data[i][j];
            }
        }

        for (int j = 0; j < numFeatures; j++) {
            mean[j] /= numSamples;
        }

        return mean;
    }

    // Step 2: Calculate the covariance matrix
    private void calculateCovarianceMatrix() {
        int numSamples = data.length;
        int numFeatures = data[0].length;
        float[][] centeredData = new float[numSamples][numFeatures];

        // Center the data by subtracting the mean
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                centeredData[i][j] = data[i][j] - mean[j];
            }
        }

        data = null;
        mean = null;
        System.gc();

        // Compute the covariance matrix
        covarianceMatrix = new float[numFeatures][numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0.0f;
                for (int k = 0; k < numSamples; k++) {
                    sum += centeredData[k][i] * centeredData[k][j];
                }
                covarianceMatrix[i][j] = sum / (numSamples - 1);
                covarianceMatrix[j][i] = covarianceMatrix[i][j]; // Symmetric matrix
            }
        }
    }

    // Step 3: Calculate eigenvalues and eigenvectors using a simple power iteration
    private void calculateEigen() {
        int numFeatures = covarianceMatrix.length;
        eigenvalues = new float[numFeatures];
        eigenvectors = new float[numFeatures][numFeatures];

        for (int i = 0; i < numFeatures; i++) {
            float[] vector = new float[numFeatures];
            vector[i] = 1.0f; // Start with a unit vector
            eigenvalues[i] = powerIteration(covarianceMatrix, vector, 1000);
            eigenvectors[i] = vector;
        }

        // Sort eigenvalues and corresponding eigenvectors in descending order
        sortEigen();
    }

    // Power iteration method to find the dominant eigenvalue and corresponding eigenvector
    private float powerIteration(float[][] matrix, float[] vector, int maxIterations) {
        int size = vector.length;
        float[] tempVector = new float[size];
        float eigenvalue = 0.0f;

        for (int iter = 0; iter < maxIterations; iter++) {
            // Matrix-vector multiplication
            for (int i = 0; i < size; i++) {
                tempVector[i] = 0.0f;
                for (int j = 0; j < size; j++) {
                    tempVector[i] += matrix[i][j] * vector[j];
                }
            }

            // Normalize the vector
            float norm = 0.0f;
            for (int i = 0; i < size; i++) {
                norm += tempVector[i] * tempVector[i];
            }
            norm = (float) Math.sqrt(norm);

            for (int i = 0; i < size; i++) {
                vector[i] = tempVector[i] / norm;
            }

            // Approximate eigenvalue
            eigenvalue = 0.0f;
            for (int i = 0; i < size; i++) {
                eigenvalue += vector[i] * tempVector[i];
            }
        }

        return eigenvalue;
    }

    // Sort eigenvalues and eigenvectors in descending order
    private void sortEigen() {
        for (int i = 0; i < eigenvalues.length - 1; i++) {
            for (int j = i + 1; j < eigenvalues.length; j++) {
                if (eigenvalues[i] < eigenvalues[j]) {
                    float tempValue = eigenvalues[i];
                    eigenvalues[i] = eigenvalues[j];
                    eigenvalues[j] = tempValue;

                    float[] tempVector = eigenvectors[i];
                    eigenvectors[i] = eigenvectors[j];
                    eigenvectors[j] = tempVector;
                }
            }
        }
    }

    // Step 4: Transform the data to the new space with k principal components
    public float[][] transform(int k) {
        int numSamples = data.length;
        int numFeatures = data[0].length;
        float[][] result = new float[numSamples][k];
        float[][] centeredData = new float[numSamples][numFeatures];

        // Center the data by subtracting the mean
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                centeredData[i][j] = data[i][j] - mean[j];
            }
        }

        // Project the data onto the top k eigenvectors
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < k; j++) {
                for (int l = 0; l < numFeatures; l++) {
                    result[i][j] += centeredData[i][l] * eigenvectors[j][l];
                }
            }
        }

        return result;
    }
}