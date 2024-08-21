package com.github.retrooper.bigdata;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class DeepLearning4jMain {
    public static void main(String[] args) throws IOException {
        DataSetIterator train = new MnistDataSetIterator(100, 60000, true);
        DataSetIterator test  = new MnistDataSetIterator(100, 10000, true);
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.UNIFORM)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .nIn(784)
                        .nOut(1000)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID)
                        .nIn(1000)
                        .nOut(10)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setLearningRate(0.7);
        model.fit(train);
    }
}
