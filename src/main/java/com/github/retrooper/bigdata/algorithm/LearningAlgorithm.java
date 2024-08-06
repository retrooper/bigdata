package com.github.retrooper.bigdata.algorithm;

public interface LearningAlgorithm<T> {
    double predict(T x);
}
