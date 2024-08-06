package com.github.retrooper.bigdata.model.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.model.Model;
import com.github.retrooper.bigdata.model.ProductionModel;

import java.util.function.Supplier;

public class SupervisedTrainingModel implements Model {
    public SupervisedTrainingModel() {
    }

    public ProductionModel train(Supplier<LearningAlgorithm> supplier) {
        LearningAlgorithm algorithm = supplier.get();
        return new ProductionModel() {
            @Override
            public double predict(double x) {
                return algorithm.predict(x);
            }
        };
    }
}
