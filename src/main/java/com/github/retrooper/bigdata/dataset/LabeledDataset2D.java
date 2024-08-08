package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;

public class LabeledDataset2D implements Dataset {
    private final Map<Float, Float> data = new HashMap<>();
    public LabeledDataset2D(Float[] input, Float[] output) {
        for (int i = 0; i < input.length; i++) {
            if (output != null)
                getData().put(input[i], output[i]);
            else
                getData().put(input[i], null);
        }
    }

    public LabeledDataset2D(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            if (output != null)
                getData().put(input[i], output[i]);
            else
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
            float value = entry.getValue() != null ? entry.getValue() : 0.0F;
            float[] data = new float[] {entry.getKey(), value};
            NDimensionalPoint point = new NDimensionalPoint(data);
            if (!consumer.test(point)) break;
        }
    }

    public Map<Float, Float> getData() {
        return data;
    }
}