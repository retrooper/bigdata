package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.unsupervised.KMeansClusteringAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset2D;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;
import com.github.retrooper.bigdata.util.Point;

import java.util.Scanner;
import java.util.function.Supplier;

public class KMeansClusterMain {
    public static void main(String[] args) {
        float[] input = new float[]{
                1.1f, 1.1f, 1.1f, 1.4f, 3f, 3.2f, 3.3f, 3.4f, 5f, 5f, 5f
        };

        FunctionDataset2D function = new FunctionDataset2D(input);
        Supplier<LearningAlgorithm<Point>> dataSupplier = () -> KMeansClusteringAlgorithm.build(3, function);
        TrainingModel<Point> trainingModel = new TrainingModel<>();
        ProductionModel<Point> trainedModel = trainingModel.train(dataSupplier);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("What cluster should we put X in, good grades (1), mid grades (2), bad grades (3)");
            String line = scanner.nextLine();
            try {
                float x = (float) Double.parseDouble(line);
                System.out.println("X: " + x + " in cluster: " + trainedModel.predict(new Point(x, 0)));
            } catch (Exception exception) {
                break;
            }
        }
    }
}