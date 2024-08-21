package com.github.retrooper.bigdata.dimensionreduction;

import com.github.retrooper.bigdata.util.ArrayConversions;
import org.apache.commons.math3.linear.*;

import java.util.Arrays;
import java.util.stream.IntStream;

public class LDA {
    public float[][] data;
    public int[] labels;
    private int nComponents;
    private RealMatrix eigenVectors;

    public LDA(int nComponents) {
        this.nComponents = nComponents;
    }

    public void fit(float[][] X, int[] y) {
        int nFeatures = X[0].length;
        int nClasses = (int) Arrays.stream(y).distinct().count();

        float[] meanTotal = calculateMean(X);
        float[][] meanClasses = new float[nClasses][nFeatures];
        int[] classCounts = new int[nClasses];

        // Calculate class means and counts
        for (int i = 0; i < y.length; i++) {
            int label = y[i];
            classCounts[label]++;
            for (int j = 0; j < nFeatures; j++) {
                meanClasses[label][j] += X[i][j];
            }
        }

        for (int label = 0; label < nClasses; label++) {
            for (int j = 0; j < nFeatures; j++) {
                meanClasses[label][j] /= classCounts[label];
            }
        }

        // Calculate scatter within (Sw) and scatter between (Sb) matrices
        RealMatrix scatterWithin = new Array2DRowRealMatrix(nFeatures, nFeatures);
        RealMatrix scatterBetween = new Array2DRowRealMatrix(nFeatures, nFeatures);

        for (int i = 0; i < X.length; i++) {
            int label = y[i];
            RealMatrix xi = new Array2DRowRealMatrix(new double[][]{ ArrayConversions.convertFToD(X[i]) }).transpose();
            RealMatrix meanClass = new Array2DRowRealMatrix(new double[][]{ArrayConversions.convertFToD(meanClasses[label])}).transpose();
            RealMatrix diff = xi.subtract(meanClass);
            scatterWithin = scatterWithin.add(diff.multiply(diff.transpose()));
        }

        for (int label = 0; label < nClasses; label++) {
            RealMatrix meanClass = new Array2DRowRealMatrix(new double[][]{ArrayConversions.convertFToD(meanClasses[label])}).transpose();
            RealMatrix meanDiff = meanClass.subtract(new Array2DRowRealMatrix(new double[][]{ArrayConversions.convertFToD(meanTotal)}).transpose());
            scatterBetween = scatterBetween.add(meanDiff.multiply(meanDiff.transpose()).scalarMultiply(classCounts[label]));
        }

        // Compute the generalized eigenvalue problem for inv(Sw) * Sb
        RealMatrix scatterWithinInv = new LUDecomposition(scatterWithin).getSolver().getInverse();
        RealMatrix swsb = scatterWithinInv.multiply(scatterBetween);

        // Eigen decomposition
        EigenDecomposition eigenDecomposition = new EigenDecomposition(swsb);
        float[][] eigenVectorsMatrix = new float[nFeatures][nComponents];

        for (int i = 0; i < nComponents; i++) {
            eigenVectorsMatrix[i] = ArrayConversions.convertDToF(eigenDecomposition.getEigenvector(i).toArray());
        }

        this.eigenVectors = new Array2DRowRealMatrix(ArrayConversions.convertTwoFTwoD(eigenVectorsMatrix)).transpose();
    }

    public float[][] transform(float[][] X) {
        RealMatrix dataMatrix = new Array2DRowRealMatrix(ArrayConversions.convertTwoFTwoD(X));
        RealMatrix projectedData = dataMatrix.multiply(eigenVectors);
        double[][] data = projectedData.getData();
        return ArrayConversions.convertTwoDTwoF(data);
    }

    public float[] transformSingle(float[] x) {
        RealMatrix dataMatrix = new Array2DRowRealMatrix(new double[][] {ArrayConversions.convertFToD(x)});
        RealMatrix projectedData = dataMatrix.multiply(eigenVectors);
        return ArrayConversions.convertDToF(projectedData.getRow(0));
    }

    private float[] calculateMean(float[][] X) {
        int nFeatures = X[0].length;
        float[] mean = new float[nFeatures];
        for (float[] xi : X) {
            for (int j = 0; j < nFeatures; j++) {
                mean[j] += xi[j];
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            mean[j] /= X.length;
        }
        return mean;
    }
}

