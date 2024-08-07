package com.github.retrooper.bigdata.util;

import java.util.Arrays;

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

    public float distanceSquared(NDimensionalPoint point) {
        float distanceSquared = 0.0F;
        for (int i = 0; i < coordinates.length; i++) {
            float difference = coordinates[i] - point.coordinates[i];
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
    public boolean equals(Object obj) {
        if (!(obj instanceof NDimensionalPoint)) return false;
        return Arrays.equals(coordinates, ((NDimensionalPoint)obj).coordinates);
    }

    @Override
    public NDimensionalPoint clone() {
        return new NDimensionalPoint(coordinates());
    }

    public float[] coordinates() {
        return coordinates;
    }

}
