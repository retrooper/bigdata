package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.unsupervised.KMeansClusteringAlgorithm;
import com.github.retrooper.bigdata.dataset.UnlabeledDatasetND;
import com.github.retrooper.bigdata.image.Image;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;
import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.dimensionreduction.PCA;
import org.opencv.core.Size;

import java.io.File;
import java.util.function.Supplier;

public class ImageClassificationMain {
    public static void main(String[] args) {
        File trainingDataDir = new File("src/main/resources/training");
        File[] files = trainingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");
        PCA pca = new PCA(5);
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
            }
        }
        pca.init();
        System.out.println("Successfully read all image data!");

        UnlabeledDatasetND function = new UnlabeledDatasetND(pca.transform(2));
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
