# *bigdata*
Easy to use Machine Learning library written in Java, powered by OpenCV.

## Linear Regression Algorithm
![image](https://github.com/user-attachments/assets/b7c6386d-448c-41f1-a976-ccfd441b00e1)

### Implementing Linear-Regression
```java
        //Data samples
        float[] input = new float[]{
                -1f, 0f, 1f, 2f
        };
        //Supervised output
        float[] output = new float[]{
                -2f, 0f, 2f, 4f
        };

        //Specify the dataset
        LabeledDataset2D function = new LabeledDataset2D(input, output);
        //Specify the learning algorithm (linear regression)
        Supplier<LearningAlgorithm<Float>> dataSupplier = () -> LinearRegressionAlgorithm.build(function);
        TrainingModel<Float> trainingModel = new TrainingModel<>();
        //Train the model
        ProductionModel<Float> trainedModel = trainingModel.train(dataSupplier);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("What X value should we predict based on the data?");
            String line = scanner.nextLine();
            try {
                float x = (float)Double.parseDouble(line);
                //Predict with the model.
                System.out.println("X: " + x + ", y: " + trainedModel.predict(x));
            }
            catch (Exception exception) {
                break;
            }
        }
```

## K-Means Clustering Algorithm
![image](https://github.com/user-attachments/assets/7a105f4c-fa4b-459a-939b-5c745e031ee9)

### Implementing K-Means Clustering Algorithm
```java
        //Unsupervised data, we expect to cluster
        float[] input = new float[]{
                1.1f, 1.1f, 1.1f, 1.4f, 3f, 3.2f, 3.3f, 3.4f, 5f, 5f, 5f
        };
        //Dataset
        UnlabeledDataset1D function = new UnlabeledDataset1D(input);
        //Learning algorithm with 3 clusters (groups), with 5 iterations
        Supplier<LearningAlgorithm<NDimensionalPoint>> dataSupplier = () -> KMeansClusteringAlgorithm.build(3, function, 5);
        TrainingModel<NDimensionalPoint> trainingModel = new TrainingModel<>();
        ProductionModel<NDimensionalPoint> trainedModel = trainingModel.train(dataSupplier);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("What cluster should we put X in, good grades (1), mid grades (2), bad grades (3)");
            String line = scanner.nextLine();
            try {
                float x = (float) Double.parseDouble(line);
                //Predict with the model
                System.out.println("X: " + x + " in cluster: " + trainedModel.predict(new NDimensionalPoint(x)));
            } catch (Exception exception) {
                break;
            }
        }
    }
```
