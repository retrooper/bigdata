package com.github.retrooper.bigdata.model;

public interface Prediction<T> {
    float predict(T x);
}
