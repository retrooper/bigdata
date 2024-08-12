package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.algorithm.supervised.KNearestNeighborsAlgorithm;
import com.github.retrooper.bigdata.dataset.SimpleLabeledDatasetND;
import com.github.retrooper.bigdata.image.Image;
import com.github.retrooper.bigdata.model.ProductionModel;
import com.github.retrooper.bigdata.model.TrainingModel;
import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.util.PCA;
import org.opencv.core.Size;

import java.io.File;
import java.util.function.Supplier;

public class ImageClassificationWithKNNMain {
    public static int DATA_WIDTH = 128;
    public static int DATA_HEIGHT = 128;
    public static void main(String[] args) {
        PCA pca = training(true);
        SimpleLabeledDatasetND function = new SimpleLabeledDatasetND(pca.transform(2), pca.labels);
        //int bestK = testing(pca, function, false);
        int bestK = 19;
        int[] predictions = prediction(pca, bestK, function, true);
        //19 works best
    }

    public static PCA training(boolean debug) {
        File trainingDataDir = new File("src/main/resources/training");
        File[] files = trainingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");
        PCA pca = new PCA(5);//was 1000!
        pca.data = new float[files.length][];
        pca.labels = new int[files.length];
        Size imageSize = new Size(DATA_WIDTH, DATA_HEIGHT);
        for (int i = 0; i < files.length; i++) {
            File trainingImageFile = files[i];
            Image trainingImage = new Image(trainingImageFile.getPath());
            Size currentSize = new Size(trainingImage.width(), trainingImage.height());
            if (imageSize == null) {
                imageSize = currentSize.clone();
            } else if (imageSize.width != currentSize.width || imageSize.height != currentSize.height) {
                // Inconsistent image sizing
                System.err.println("Image sizes are inconsistent. We will resize them all to 256x256");
                for (i = 0; i < files.length; i++) {
                    trainingImage = new Image(files[i].getPath());
                    trainingImage.resize(DATA_WIDTH, DATA_HEIGHT);
                    trainingImage.save();
                }
                System.out.println("Successfully resized all images to 64x64. Please run the program again!");
                System.exit(0);
            }
            pca.data[i] = trainingImage.features().get().getData();

            // Label the data
            String imageName = trainingImage.path().toLowerCase();
            if (imageName.contains("dog")) {
                pca.labels[i] = 0;
            } else {
                pca.labels[i] = 1;
            }
        }
        pca.init();
        if (debug)
            System.out.println("Successfully processed all image data!");
        return pca;
    }

    public static int testing(PCA pca, SimpleLabeledDatasetND function, boolean debug) {
        //Test with random image: in testing
        File testingDataDir = new File("src/main/resources/testing");
        File[] files = testingDataDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");

        //Find the k value with highest accuracy in testing data
        if (debug)
            System.out.println("Testing and evaluating the model with testing data.");
        int bestK = 2;
        int maxWrongCount = files.length;
        for (int k = 2; k < 50; k++) {
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
            }

            if (wrongCount < maxWrongCount) {
                maxWrongCount = wrongCount;
                bestK = k;
            }
        }

        if (debug)
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
            if (debug)
                System.out.println("image name: " + test.path() + ", cluster: " + (value == 0 ? "dog" : "cat"));
        }

        return bestK;
    }

    public static int[] prediction(PCA pca, int k, SimpleLabeledDatasetND function, boolean debug) {
        //Predict labels for images in prediction folder
        File predictionFilesDir = new File("src/main/resources/prediction");
        File[] files = predictionFilesDir.listFiles();
        if (files == null) throw new IllegalStateException("Failed to find training data");


        Supplier<LearningAlgorithm<NDimensionalPoint>> dataSupplier =
                () -> KNearestNeighborsAlgorithm.build(k, function);
        TrainingModel<NDimensionalPoint> trainingModel = new TrainingModel<>();
        ProductionModel<NDimensionalPoint> trainedModel = trainingModel.train(dataSupplier);

        int[] output = new int[files.length];
        for (int i = 0; i < files.length; i++) {
            File newImageFile = files[i];
            Image test = new Image(newImageFile.getPath());
            float[] data = test.features().get().getData();
            NDimensionalPoint point = new NDimensionalPoint(pca.transformSingleSample(data, 2));
            float value = trainedModel.predict(point);
            output[i] = (int) value;
            if (debug)
                System.out.println("Prediction of Image: " + test.path() + ", Label: " + (value == 0 ? "dog" : "cat"));
        }
        return output;
    }
}
