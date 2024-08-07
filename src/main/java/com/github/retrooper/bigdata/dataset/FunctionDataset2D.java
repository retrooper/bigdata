package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.Point;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Stream;

public class FunctionDataset2D<T extends Number, Z extends Number> implements Dataset<T, Z> {
    private final Map<T, Z> data = new HashMap<>();
    public FunctionDataset2D(T[] input, Z[] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public int dataPoints() {
        return getData().keySet().size();
    }

    public void iteratePoints(Predicate<Point> consumer) {
        for (Map.Entry<T, Z> entry : getData().entrySet()) {
            Point point = new Point(entry.getKey().doubleValue(), entry.getValue().doubleValue());
            if (!consumer.test(point)) break;
        }
    }

    public Map<T, Z> getData() {
        return data;
    }
}