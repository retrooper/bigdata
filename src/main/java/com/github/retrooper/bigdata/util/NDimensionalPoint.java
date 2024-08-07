package com.github.retrooper.bigdata.util;

import java.util.stream.Stream;

public class NDimensionalPoint {
    private final float[] coordinates;

    public NDimensionalPoint(float singularCoord) {
        this(new float[]{singularCoord});
    }

    public NDimensionalPoint(float[] coordinates) {
        this.coordinates = coordinates;
    }

    public NDimensionalPoint(Float[] coordinates) {
        float[] array = new float[coordinates.length];
        for (int i = 0; i < array.length; i++) {
            array[i] = coordinates[i];
        }
        this.coordinates = array;
    }

    public double distanceSquared(NDimensionalPoint point) {
        float distanceSquared = 0.0F;
        for (int i = 0; i < coordinates.length; i++) {
            double difference = coordinates[i] - point.coordinates[i];
            distanceSquared += difference * difference;
        }
        return distanceSquared;
    }

    public float distance(NDimensionalPoint point) {
        return (float) Math.sqrt(distanceSquared(point));
    }

    public float getCoordinatesSum() {
        float sum = 0.0F;
        for (float c : coordinates) {
            sum += c;
        }
        return sum;
    }

    @Override
    public NDimensionalPoint clone() {
        return new NDimensionalPoint(coordinates());
    }

    public float[] coordinates() {
        return coordinates;
    }

}
