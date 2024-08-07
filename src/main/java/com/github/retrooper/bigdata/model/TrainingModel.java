package com.github.retrooper.bigdata.model;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;

import java.util.function.Supplier;

public class TrainingModel<T> implements Model {
    public TrainingModel() {
    }

    public ProductionModel<T> train(Supplier<LearningAlgorithm<T>> supplier) {
        LearningAlgorithm<T> algorithm = supplier.get();
        return new ProductionModel<T>() {
            @Override
            public double predict(T x) {
                return algorithm.predict(x);
            }
        };
    }
}
