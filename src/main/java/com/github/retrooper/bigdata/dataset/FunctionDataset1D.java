package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.Point;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

public class FunctionDataset1D implements Dataset {
    private final List<Double> data = new ArrayList<>();
    public FunctionDataset1D(Double[] input) {
        for (Double aDouble : input) {
            getData().put(aDouble, 0.0);
        }
    }

    public FunctionDataset1D(double[] input) {
        for (double v : input) {
            getData().put(v, 0.0);
        }
    }

    public int dataPoints() {
        return getData().size();
    }

    public void iteratePoints(Predicate<Double> consumer) {
        for (double x : getData()) {
            if (!consumer.test(x)) break;
        }
    }

    public List<Double> getData() {
        return data;
    }
}