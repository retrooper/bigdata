package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.Point;

import java.util.Map;
import java.util.function.Predicate;

public class FunctionDataset<T extends Number, Z extends Number> extends Dataset<T, Z> {
    public FunctionDataset(T[] input, Z[] output) {
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
}