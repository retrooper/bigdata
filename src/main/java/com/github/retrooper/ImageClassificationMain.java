package com.github.retrooper;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.unsupervised.KMeansClusteringAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDatasetNDimensional;
import com.github.retrooper.bigdata.image.Image;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;
import com.github.retrooper.bigdata.util.Point;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

public class ImageClassificationMain {
    public static void main(String[] args) {
        File trainingDataDir = new File("src/main/resources/training");
        File[] files = trainingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");
        List<Image> trainingImages = new ArrayList<>(files.length);
        for (File trainingImageFile : files) {
            trainingImages.add(new Image(trainingImageFile.getPath()));
        }


        int maxFeatures = Integer.MAX_VALUE;
        for (Image t : trainingImages) {
            double[] data = t.features().get().getData();
            System.out.println("feature count: " + data.length);
            if (data.length < maxFeatures) {
                maxFeatures = data.length;
            }
        }

        Double[][] inputData = new Double[trainingImages.size()][];

        for (int i = 0; i < trainingImages.size(); i++) {
            double[] rawData = trainingImages.get(i).features().get().getData();

            double[] data = new double[maxFeatures];
            System.arraycopy(rawData, 0, data, 0, data.length);
            inputData[i] = Arrays.stream(data).boxed().toArray(Double[]::new);
        }

        Double[][] outputData = new Double[inputData.length][];
        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = new Double[]{0.0};
        }

        /*FunctionDatasetNDimensional<Double, Double> function = new FunctionDatasetNDimensional<>(inputData, outputData);
        Supplier<LearningAlgorithm<Point>> dataSupplier = () -> KMeansClusteringAlgorithm.build(3, function);
        TrainingModel<Point> trainingModel = new TrainingModel<>();
        ProductionModel<Point> trainedModel = trainingModel.train(dataSupplier);*/

    }
}
