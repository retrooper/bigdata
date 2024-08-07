package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.util.Point;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Stream;

public class FunctionDataset2D implements Dataset {
    private final Map<Double, Double> data = new HashMap<>();
    public FunctionDataset2D(Double[] input, Double[] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public FunctionDataset2D(double[] input, double[] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    @Override
    public int dataPoints() {
        return getData().keySet().size();
    }

    @Override
    public void iteratePoints(Predicate<NDimensionalPoint> consumer) {
        for (Map.Entry<Double, Double> entry : getData().entrySet()) {
            Point point = new Point(entry.getKey(), entry.getValue());
            if (!consumer.test(point)) break;
        }
    }

    public Map<Double, Double> getData() {
        return data;
    }
}