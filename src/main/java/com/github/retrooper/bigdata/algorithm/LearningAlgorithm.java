package com.github.retrooper.bigdata.algorithm;

public interface LearningAlgorithm<T> {
    float predict(T x);
}
