package com.github.retrooper.bigdata.model;

public interface Prediction<T> {
    double predict(T x);
}
