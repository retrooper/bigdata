package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;
import org.apache.commons.lang3.ArrayUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;

public class UnlabeledDatasetND implements Dataset {
    private final Map<Float[], Float[]> data = new HashMap<>();
    public UnlabeledDatasetND(Float[][] input, Float[][] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public UnlabeledDatasetND(float[][] input, float[][] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(ArrayUtils.toObject(input[i]), ArrayUtils.toObject(output[i]));
        }
    }

    public UnlabeledDatasetND(Float[][] input) {
        for (Float[] floats : input) {
            getData().put(floats, null);
        }
    }


    // Memory intensive
    @Deprecated
    public UnlabeledDatasetND(float[][] input) {
        for (float[] array : input) {
            getData().put(ArrayUtils.toObject(array), null);
        }
    }


    @Override
    public int dataPoints() {
        return getData().keySet().size();
    }

    @Override
    public void iteratePoints(Predicate<NDimensionalPoint> consumer) {
        for (Map.Entry<Float[], Float[]> entry : getData().entrySet()) {

            NDimensionalPoint point = new NDimensionalPoint(entry.getKey());
            if (!consumer.test(point)) break;
        }
    }

    public Map<Float[], Float[]> getData() {
        return data;
    }
}