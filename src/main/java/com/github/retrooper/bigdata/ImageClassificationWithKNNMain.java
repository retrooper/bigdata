package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.supervised.KNearestNeighborsAlgorithm;
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
        System.out.println("Successfully processed all image data!");
        LabeledDatasetND function = new LabeledDatasetND(pca.transform(2), output);

        //Test with random image: in testing
        File testingDataDir = new File("src/main/resources/testing");
        files = testingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");

        //Find the k value with highest accuracy in testing data
        System.out.println("Testing and evaluating the model with testing data.");
        int bestK = 2;
        int maxWrongCount = files.length;
        for (int k = 2; k < 100; k++) {
            int finalK = k;
            Supplier<LearningAlgorithm<NDimensionalPoint>> dataSupplier =
                    () -> KNearestNeighborsAlgorithm.build(finalK, function);
            TrainingModel<NDimensionalPoint> trainingModel = new TrainingModel<>();
            ProductionModel<NDimensionalPoint> trainedModel = trainingModel.train(dataSupplier);

            int wrongCount = 0;
            for (File testingImageFile : files) {
                Image test = new Image(testingImageFile.getPath());
                float[] data = test.features().get().getData();
                NDimensionalPoint point = new NDimensionalPoint(pca.transformSingleSample(data, 2));
                float value = trainedModel.predict(point);
                if (value != 0 && test.path().contains("dog")
                || value != 1 && test.path().contains("cat")) {
                    wrongCount++;
                }
                //System.out.println("image name: " + test.path() + ", cluster: " + (trainedModel.predict(point) == 0 ? "dog" : "cat"));
            }

            if (wrongCount < maxWrongCount) {
                maxWrongCount = wrongCount;
                bestK = k;
            }
        }

        System.out.println("After evaluating with the testing data, we found the best k value to be: " + bestK + ", with " + maxWrongCount + " mistakes.");
        int finalBestK = bestK;
        Supplier<LearningAlgorithm<NDimensionalPoint>> dataSupplier =
                () -> KNearestNeighborsAlgorithm.build(finalBestK, function);
        TrainingModel<NDimensionalPoint> trainingModel = new TrainingModel<>();
        ProductionModel<NDimensionalPoint> trainedModel = trainingModel.train(dataSupplier);

        for (File testingImageFile : files) {
            Image test = new Image(testingImageFile.getPath());
            float[] data = test.features().get().getData();
            NDimensionalPoint point = new NDimensionalPoint(pca.transformSingleSample(data, 2));
            float value = trainedModel.predict(point);
            System.out.println("image name: " + test.path() + ", cluster: " + (value == 0 ? "dog" : "cat"));
        }

        //14 works best
    }
}
