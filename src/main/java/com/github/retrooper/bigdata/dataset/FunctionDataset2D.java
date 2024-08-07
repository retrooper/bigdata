package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.util.Point;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Stream;

public class FunctionDataset2D implements Dataset {
    private final Map<Float, Float> data = new HashMap<>();
    public FunctionDataset2D(Float[] input, Float[] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public FunctionDataset2D(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public FunctionDataset2D(Float[] input) {
        for (Float aFloat : input) {
            getData().put(aFloat, null);
        }
    }

    public FunctionDataset2D(float[] input) {
        for (float v : input) {
            getData().put(v, null);
        }
    }

    @Override
    public int dataPoints() {
        return getData().keySet().size();
    }

    @Override
    public void iteratePoints(Predicate<NDimensionalPoint> consumer) {
        for (Map.Entry<Float, Float> entry : getData().entrySet()) {
            Point point = new Point(entry.getKey(), entry.getValue());
            if (!consumer.test(point)) break;
        }
    }

    public Map<Float, Float> getData() {
        return data;
    }
}