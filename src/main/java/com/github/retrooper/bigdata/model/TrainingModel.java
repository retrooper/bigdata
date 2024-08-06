package com.github.retrooper.bigdata.model;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;

import java.util.function.Supplier;

public class TrainingModel implements Model {
    public TrainingModel() {
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
