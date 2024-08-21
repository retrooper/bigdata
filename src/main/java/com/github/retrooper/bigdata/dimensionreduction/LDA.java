package com.github.retrooper.bigdata.dimensionreduction;

import com.github.retrooper.bigdata.util.ArrayConversions;
import org.apache.commons.math3.linear.*;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class LDA {
    public float[][] totalData;
    public int[] totalLabels;
    private int nComponents;
    private RealMatrix eigenVectors;
    private RealMatrix totalMean;
    private RealMatrix Sw;
    private RealMatrix Sb;
    private Map<Integer, RealMatrix> classMeans;
    private Map<Integer, Integer> classCounts;
    private int nFeatures;
    private int totalSamples;

    public LDA(int nComponents, int nFeatures) {
        this.nComponents = nComponents;
        this.nFeatures = nFeatures;
        this.Sw = new Array2DRowRealMatrix(nFeatures, nFeatures);
        this.Sb = new Array2DRowRealMatrix(nFeatures, nFeatures);
        this.classMeans = new HashMap<>();
        this.classCounts = new HashMap<>();
        this.totalMean = new Array2DRowRealMatrix(nFeatures, 1);
        this.totalSamples = 0;
    }

    public void partialFit(float[][] X, int[] y) {
        int batchSize = X.length;

        // Initialize batch statistics
        Map<Integer, RealMatrix> batchClassMeans = new HashMap<>();
        Map<Integer, Integer> batchClassCounts = new HashMap<>();

        // Accumulate sums for each class
        for (int i = 0; i < batchSize; i++) {
            int label = y[i];
            RealMatrix xi = new Array2DRowRealMatrix(new double[][]{ArrayConversions.convertFToD(X[i])}).transpose();

            batchClassMeans.putIfAbsent(label, new Array2DRowRealMatrix(nFeatures, 1));
            batchClassCounts.putIfAbsent(label, 0);

            RealMatrix currentBatchMean = batchClassMeans.get(label);
            RealMatrix updatedBatchMean = currentBatchMean.add(xi);
            batchClassMeans.put(label, updatedBatchMean);
            batchClassCounts.put(label, batchClassCounts.get(label) + 1);
        }

        // Update class means and scatter matrices
        for (Map.Entry<Integer, RealMatrix> entry : batchClassMeans.entrySet()) {
            int label = entry.getKey();
            RealMatrix batchMean = entry.getValue().scalarMultiply(1.0 / batchClassCounts.get(label));
            RealMatrix oldMean = classMeans.getOrDefault(label, new Array2DRowRealMatrix(nFeatures, 1));
            int oldCount = classCounts.getOrDefault(label, 0);
            int newCount = oldCount + batchClassCounts.get(label);

            // Update class mean
            RealMatrix newMean = oldMean.scalarMultiply(oldCount).add(batchMean.scalarMultiply(batchClassCounts.get(label))).scalarMultiply(1.0 / newCount);
            classMeans.put(label, newMean);
            classCounts.put(label, newCount);

            // Update within-class scatter (Sw)
            for (int i = 0; i < X.length; i++) {
                if (y[i] == label) {
                    RealMatrix xi = new Array2DRowRealMatrix(new double[][]{ArrayConversions.convertFToD(X[i])}).transpose();
                    RealMatrix diff = xi.subtract(newMean);
                    Sw = Sw.add(diff.multiply(diff.transpose()));
                }
            }

            // Update global mean
            RealMatrix classMeanSum = totalMean.scalarMultiply(totalSamples).add(batchMean.scalarMultiply(batchClassCounts.get(label)));
            totalMean = classMeanSum.scalarMultiply(1.0 / (totalSamples + batchSize));

            // Update between-class scatter (Sb)
            RealMatrix meanDiff = newMean.subtract(totalMean);
            Sb = Sb.add(meanDiff.multiply(meanDiff.transpose()).scalarMultiply(classCounts.get(label)));
        }

        totalSamples += batchSize;
    }

    public void finalizeModel() {
        RealMatrix scatterWithinInv = new LUDecomposition(Sw).getSolver().getInverse();
        RealMatrix swsb = scatterWithinInv.multiply(Sb);

        EigenDecomposition eigenDecomposition = new EigenDecomposition(swsb);
        double[][] eigenVectorsMatrix = new double[nFeatures][nComponents];

        for (int i = 0; i < nComponents; i++) {
            eigenVectorsMatrix[i] = eigenDecomposition.getEigenvector(i).toArray();
        }

        this.eigenVectors = new Array2DRowRealMatrix(eigenVectorsMatrix).transpose();
    }

    public float[][] transform(float[][] X) {
        RealMatrix dataMatrix = new Array2DRowRealMatrix(ArrayConversions.convertTwoFTwoD(X));
        RealMatrix projectedData = dataMatrix.multiply(eigenVectors);
        return ArrayConversions.convertTwoDTwoF(projectedData.getData());
    }

    public float[] transformSingle(float[] x) {
        RealMatrix dataMatrix = new Array2DRowRealMatrix(new double[][]{ArrayConversions.convertFToD(x)});
        RealMatrix projectedData = dataMatrix.multiply(eigenVectors);
        return ArrayConversions.convertDToF(projectedData.getRow(0));
    }
}

