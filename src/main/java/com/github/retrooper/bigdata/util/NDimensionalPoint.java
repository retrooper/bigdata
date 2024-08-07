package com.github.retrooper.bigdata.util;

public class NDimensionalPoint<T extends Number> {
    private final T[] coordinates;
    public NDimensionalPoint(T[] coordinates) {
        this.coordinates = coordinates;
    }

    public double distanceSquared(NDimensionalPoint<T> point) {
        double distanceSquared = 0;
        for (int i = 0; i < coordinates.length; i++) {
            double difference = coordinates[i].doubleValue() - point.coordinates[i].doubleValue();
            distanceSquared += difference * difference;
        }
        return distanceSquared;
    }

    public double distance(NDimensionalPoint<T> point) {
        return Math.sqrt(distanceSquared(point));
    }

    public double getCoordinatesSum() {
        double sum = 0;
        for (T c : coordinates) {
            sum += c.doubleValue();
        }
        return sum;
    }
    @Override
    public NDimensionalPoint<T> clone() {
        return new NDimensionalPoint<T>(coordinates());
    }

    public T[] coordinates() {
        return coordinates;
    }

}
