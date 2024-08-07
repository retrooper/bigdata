package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.util.*;
import java.util.function.Predicate;

public class FunctionDatasetNDimensional<T extends Number, Z extends Number> implements Dataset<T, Z> {
    private final Map<T[], Z[]> data = new HashMap<>();
    public FunctionDatasetNDimensional(T[][] input, Z[][] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public int dataPoints() {
        return getData().keySet().size();
    }

    public void iteratePoints(Predicate<NDimensionalPoint<T>> consumer) {
        for (Map.Entry<T[], Z[]> entry : getData().entrySet()) {

            NDimensionalPoint<T> point = new NDimensionalPoint<>(entry.getKey());
            if (!consumer.test(point)) break;
        }
    }

    public Map<T[], Z[]> getData() {
        return data;
    }
}