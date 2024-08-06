package com.github.retrooper.bigdata.model.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.model.Model;
import com.github.retrooper.bigdata.model.ProductionModel;

import java.util.function.Supplier;

public class SupervisedTrainingModel<T> implements Model {
    public SupervisedTrainingModel() {
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
