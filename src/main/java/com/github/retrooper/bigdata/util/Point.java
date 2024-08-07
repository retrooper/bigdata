package com.github.retrooper.bigdata.util;

public record Point(double x, double y) {
    public double distanceSquared(Point point) {
        double xDiff = x() - point.x();
        double yDiff = y() - point.y();
        return xDiff * xDiff + yDiff * yDiff;
    }

    public double distance(Point point) {
        return Math.sqrt(distanceSquared(point));
    }

    @Override
    public Point clone() {
        return new Point(x(), y());
    }
}
