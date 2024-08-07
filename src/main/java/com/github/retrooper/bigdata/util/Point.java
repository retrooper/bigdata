package com.github.retrooper.bigdata.util;

public class Point extends NDimensionalPoint {
    public Point(float x, float y) {
        super(new Float[] {x, y});
    }

    @Override
    public Point clone() {
        return new Point(coordinates()[0], coordinates()[1]);
    }

    public float x() {
        return coordinates()[0];
    }

    public float y() {
        return coordinates()[1];
    }
}
