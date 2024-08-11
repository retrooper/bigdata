package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.unsupervised.KMeansClusteringAlgorithm;
import com.github.retrooper.bigdata.dataset.LabeledDatasetND;
import com.github.retrooper.bigdata.dataset.UnlabeledDatasetND;
import com.github.retrooper.bigdata.image.Image;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;
import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.util.PCA;
import org.opencv.core.Size;

import java.io.File;
import java.util.function.Supplier;

public class ImageClassificationWithKNNMain {
    public static void main(String[] args) {
        File trainingDataDir = new File("src/main/resources/training");
        File[] files = trainingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");
        int[] output = new int[files.length];
        PCA pca = new PCA();
        {
            pca.data = new float[files.length][];
            Size imageSize = new Size(128, 128);
            for (int i = 0; i < files.length; i++) {
                File trainingImageFile = files[i];
                Image trainingImage = new Image(trainingImageFile.getPath());
                Size currentSize = new Size(trainingImage.width(), trainingImage.height());
                if (imageSize == null) {
                    imageSize = currentSize.clone();
                } else if (imageSize.width != currentSize.width || imageSize.height != currentSize.height) {
                    // Inconsistent image sizing
                    System.out.println("Image sizes are inconsistent. We will resize them all to 256x256");
                    for (i = 0; i < files.length; i++) {
                        trainingImage = new Image(files[i].getPath());
                        trainingImage.resize(128, 128);
                        trainingImage.save();
                    }
                    System.out.println("Successfully resized all images to 64x64. Please run the program again!");
                    System.exit(0);
                }
                pca.data[i] = trainingImage.features().get().getData();

                // Store its cluster
                if (trainingImage.path().contains("dog")) {
                    output[i] = 0;
                }
                else {
                    output[i] = 1;
                }
            }
        }
        pca.init();
        System.out.println("Successfully read all image data!");
        System.out.println("data size; " + pca.data.length + ", pca size: " + pca.transform(2).length);
        LabeledDatasetND function = new LabeledDatasetND(pca.transform(2), output);
        Supplier<LearningAlgorithm<NDimensionalPoint>> dataSupplier =
                () -> KMeansClusteringAlgorithm.build(2, function, 50);
        TrainingModel<NDimensionalPoint> trainingModel = new TrainingModel<>();
        ProductionModel<NDimensionalPoint> trainedModel = trainingModel.train(dataSupplier);

        //Test with random image: in testing
        File testingDataDir = new File("src/main/resources/testing");
        files = testingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");

        for (File testingImageFile : files) {
            Image test = new Image(testingImageFile.getPath());
            float[] data = test.features().get().getData();
            NDimensionalPoint point = new NDimensionalPoint(pca.transformSingleSample(data, 2));
            System.out.println("image name: " + test.path() + ", cluster: " + trainedModel.predict(point));
        }
    }
}
