package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.util.Point;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

public class FunctionDataset1D implements Dataset {
    private final List<NDimensionalPoint> data = new ArrayList<>();
    public FunctionDataset1D(Double[] input) {
        for (double v : input) {
            getData().add(new NDimensionalPoint(v));
        }
    }

    public FunctionDataset1D(double[] input) {
        for (double v : input) {
            getData().add(new NDimensionalPoint(v));
        }
    }

    @Override
    public int dataPoints() {
        return getData().size();
    }

    public void iteratePoints(Predicate<NDimensionalPoint> consumer) {
        for (double x : getData()) {
            NDimensionalPoint p = new NDimensionalPoint(x);
            if (!consumer.test(p)) break;
        }
    }

    public List<Double> getData() {
        return data;
    }
}