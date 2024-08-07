package com.github.retrooper.bigdata.util;

import java.util.stream.Stream;

public class NDimensionalPoint {
    private final double[] coordinates;

    public NDimensionalPoint(double singularCoord) {
        this(new double[]{singularCoord});
    }

    public NDimensionalPoint(double[] coordinates) {
        this.coordinates = coordinates;
    }

    public NDimensionalPoint(Double[] coordinates) {
        double[] array = new double[coordinates.length];
        for (int i = 0; i < array.length; i++) {
            array[i] = coordinates[i];
        }
        this.coordinates = array;
    }

    public double distanceSquared(NDimensionalPoint point) {
        double distanceSquared = 0;
        for (int i = 0; i < coordinates.length; i++) {
            double difference = coordinates[i] - point.coordinates[i];
            distanceSquared += difference * difference;
        }
        return distanceSquared;
    }

    public double distance(NDimensionalPoint point) {
        return Math.sqrt(distanceSquared(point));
    }

    public double getCoordinatesSum() {
        double sum = 0;
        for (double c : coordinates) {
            sum += c;
        }
        return sum;
    }

    @Override
    public NDimensionalPoint clone() {
        return new NDimensionalPoint(coordinates());
    }

    public double[] coordinates() {
        return coordinates;
    }

}
