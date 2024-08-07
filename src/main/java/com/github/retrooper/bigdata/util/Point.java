package com.github.retrooper.bigdata.util;

public class Point extends NDimensionalPoint {
    public Point(double x, double y) {
        super(new Double[] {x, y});
    }

    @Override
    public Point clone() {
        return new Point(coordinates()[0], coordinates()[1]);
    }

    public double x() {
        return coordinates()[0];
    }

    public double y() {
        return coordinates()[1];
    }
}
