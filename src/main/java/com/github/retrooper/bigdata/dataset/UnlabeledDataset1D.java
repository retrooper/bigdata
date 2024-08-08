package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;

public class UnlabeledDataset1D implements Dataset {
    private final Map<Float, Float> data = new HashMap<>();
    public UnlabeledDataset1D(Float[] input) {
        for (int i = 0; i < input.length; i++) {
                getData().put(input[i], null);
        }
    }

    public UnlabeledDataset1D(float[] input) {
        for (int i = 0; i < input.length; i++) {
                getData().put(input[i], null);
        }
    }

    @Override
    public int dataPoints() {
        return getData().keySet().size();
    }

    @Override
    public void iteratePoints(Predicate<NDimensionalPoint> consumer) {
        for (Map.Entry<Float, Float> entry : getData().entrySet()) {
            NDimensionalPoint point = new NDimensionalPoint(new float[]{entry.getKey()});
            if (!consumer.test(point)) break;
        }
    }

    public Map<Float, Float> getData() {
        return data;
    }
}