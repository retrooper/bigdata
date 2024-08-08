package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.supervised.LinearRegressionAlgorithm;
import com.github.retrooper.bigdata.dataset.LabeledDataset2D;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;

import java.util.Scanner;
import java.util.function.Supplier;

public class LinearRegressionMain {
    public static void main(String[] args) {
        float[] input = new float[]{
                -1f, 0f, 1f, 2f
        };

        float[] output = new float[]{
                -2f, 0f, 2f, 4f
        };

        LabeledDataset2D function = new LabeledDataset2D(input, output);
        Supplier<LearningAlgorithm<Float>> dataSupplier = () -> LinearRegressionAlgorithm.build(function);
        TrainingModel<Float> trainingModel = new TrainingModel<>();
        ProductionModel<Float> trainedModel = trainingModel.train(dataSupplier);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("What X value should we predict based on the data?");
            String line = scanner.nextLine();
            try {
                float x = (float)Double.parseDouble(line);
                System.out.println("X: " + x + ", y: " + trainedModel.predict(x));
            }
            catch (Exception exception) {
                break;
            }
        }
    }
}