package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;
import org.apache.commons.lang3.ArrayUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Predicate;

public class LabeledDatasetND implements Dataset {
    private final Map<Float[], Integer> data = new HashMap<>();
    public LabeledDatasetND(Float[][] input, Integer[] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public LabeledDatasetND(float[][] input, int[] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(ArrayUtils.toObject(input[i]), output[i]);
        }
    }


    @Override
    public int dataPoints() {
        return getData().keySet().size();
    }

    @Override
    public void iteratePoints(Predicate<NDimensionalPoint> consumer) {
        for (Map.Entry<Float[], Integer> entry : getData().entrySet()) {

            NDimensionalPoint point = new NDimensionalPoint(entry.getKey());
            if (!consumer.test(point)) break;
        }
    }

    public void iterate(BiFunction<NDimensionalPoint, Integer, Boolean> function) {
        for (Map.Entry<Float[], Integer> entry : getData().entrySet()) {

            NDimensionalPoint point = new NDimensionalPoint(entry.getKey());
            if (!function.apply(point, entry.getValue())) break;
        }
    }

    public Map<Float[], Integer> getData() {
        return data;
    }
}