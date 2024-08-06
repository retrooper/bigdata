package com.github.retrooper;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.LinearRegressionAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset;
import com.github.retrooper.bigdata.model.Model;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;

import java.util.Arrays;
import java.util.Scanner;
import java.util.function.Supplier;
import java.util.stream.Stream;

public class Main {
    public static void main(String[] args) {
        double[] input = new double[]{
                -1, 0, 1, 2
        };

        double[] output = new double[]{
                -2, 0, 2, 4
        };

        Double[] doubles = Arrays.stream(input).boxed().toArray(Double[]::new);

        FunctionDataset<Double, Double> function = new FunctionDataset<>(Arrays.stream(input).boxed().toArray(Double[]::new),
                Arrays.stream(output).boxed().toArray(Double[]::new));
        Supplier<LearningAlgorithm> dataSupplier = () -> LinearRegressionAlgorithm.build(function);
        TrainingModel trainingModel = new TrainingModel();
        ProductionModel trainedModel = trainingModel.train(dataSupplier);

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