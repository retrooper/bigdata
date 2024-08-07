package com.github.retrooper;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.unsupervised.KMeansClusteringAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset1D;
import com.github.retrooper.bigdata.dataset.FunctionDataset2D;
import com.github.retrooper.bigdata.dataset.FunctionDatasetNDimensional;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;
import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.util.Point;

import java.util.Arrays;
import java.util.Scanner;
import java.util.function.Supplier;

public class KMeansClusterMain {
    public static void main(String[] args) {
        double[] input = new double[]{
                1.1, 1.1, 1.1, 1.4, 3, 3.2, 3.3, 3.4, 5, 5, 5
        };

        FunctionDataset1D function = new FunctionDataset1D(input);
        Supplier<LearningAlgorithm<NDimensionalPoint>> dataSupplier = () -> KMeansClusteringAlgorithm.build(3, function);
        TrainingModel<NDimensionalPoint> trainingModel = new TrainingModel<>();
        ProductionModel<NDimensionalPoint> trainedModel = trainingModel.train(dataSupplier);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("What cluster should we put X in, good grades (1), mid grades (2), bad grades (3)");
            String line = scanner.nextLine();
            try {
                double x = Double.parseDouble(line);
                System.out.println("X: " + x + " in cluster: " + trainedModel.predict(new NDimensionalPoint(x)));
            } catch (Exception exception) {
                break;
            }
        }
    }
}