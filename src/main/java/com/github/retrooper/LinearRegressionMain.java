package com.github.retrooper;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.supervised.LinearRegressionAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset2D;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;

import java.util.Arrays;
import java.util.Scanner;
import java.util.function.Supplier;

public class LinearRegressionMain {
    public static void main(String[] args) {
        double[] input = new double[]{
                -1, 0, 1, 2
        };

        double[] output = new double[]{
                -2, 0, 2, 4
        };

        FunctionDataset2D function = new FunctionDataset2D(input, output);
        Supplier<LearningAlgorithm<Double>> dataSupplier = () -> LinearRegressionAlgorithm.build(function);
        TrainingModel<Double> trainingModel = new TrainingModel<>();
        ProductionModel<Double> trainedModel = trainingModel.train(dataSupplier);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("What X value should we predict based on the data?");
            String line = scanner.nextLine();
            try {
                double x = Double.parseDouble(line);
                System.out.println("X: " + x + ", y: " + trainedModel.predict(x));
            }
            catch (Exception exception) {
                break;
            }
        }
    }
}