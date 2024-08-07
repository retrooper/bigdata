package com.github.retrooper;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.unsupervised.KMeansClusteringAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDatasetNDimensional;
import com.github.retrooper.bigdata.image.Image;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;
import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.io.File;
import java.util.function.Supplier;

public class ImageClassificationMain {
    public static void main(String[] args) {
        File trainingDataDir = new File("src/main/resources/training");
        File[] files = trainingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");
        double[][] inputData = new double[files.length][];
        for (int i = 0; i < files.length; i++) {
            File trainingImageFile = files[i];
            Image trainingImage = new Image(trainingImageFile.getPath());
            inputData[i] = trainingImage.features().get().getData();

        }

        FunctionDatasetNDimensional function = new FunctionDatasetNDimensional(inputData);
        Supplier<LearningAlgorithm<NDimensionalPoint>> dataSupplier = () -> KMeansClusteringAlgorithm.build(2, function);
        TrainingModel<NDimensionalPoint> trainingModel = new TrainingModel<>();
        ProductionModel<NDimensionalPoint> trainedModel = trainingModel.train(dataSupplier);

        //Test with random image: in testing
        File testingDataDir = new File("src/main/resources/testing");
        files = testingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");


        for (File testingImageFile : files) {
            Image test = new Image(testingImageFile.getPath());
            double[] data = test.features().get().getData();
            NDimensionalPoint point = new NDimensionalPoint(data);

            System.out.println("image name: " + test.path() + ", cluster: " + trainedModel.predict(point));
        }
    }
}
