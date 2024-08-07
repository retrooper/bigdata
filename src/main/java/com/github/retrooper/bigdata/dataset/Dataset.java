package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.util.Point;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;

public interface Dataset {
    int dataPoints();

    void iteratePoints(Predicate<NDimensionalPoint> consumer);
}
