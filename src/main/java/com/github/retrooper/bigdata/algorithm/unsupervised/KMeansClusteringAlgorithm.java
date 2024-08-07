package com.github.retrooper.bigdata.algorithm.unsupervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset;
import com.github.retrooper.bigdata.util.Point;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class KMeansClusteringAlgorithm implements LearningAlgorithm<Point> {
    private final int k;
    private final List<Cluster> clusters;

    private KMeansClusteringAlgorithm(int k, List<Cluster> clusters) {
        this.k = k;
        this.clusters = clusters;
    }

    private static void iteration(int k, List<Cluster> clusters, FunctionDataset<Double, Double> function) {
        function.iteratePoints(point -> {
            // Find cluster with center closest to point
            Cluster cluster = Cluster.findCluster(k, clusters, point);

            // Add the point to the best fitting cluster.
            cluster.points().add(point);
            return true;
        });

        for (Cluster cluster : clusters) {
            int n = cluster.points().size();
            double xSum = 0;
            double ySum = 0;
            for (int i = 0; i < n; i++) {
                Point point = cluster.points().get(i);
                xSum += point.x();
                ySum += point.y();
            }

            // New center is the mean of all points in that particular cluster
            cluster.center(new Point(xSum / n, ySum / n));
        }
    }

    public static KMeansClusteringAlgorithm build(int k, FunctionDataset<Double, Double> function) {
        List<Cluster> clusters = new ArrayList<>(k);

        function.iteratePoints(point -> {
            clusters.add(new Cluster(point));
            // Condition to continue iterating
            return clusters.size() != k;
        });

        for (int i = 0; i < 5; i++) {
            for (Cluster c : clusters) {
                c.points().clear();
            }
            iteration(k, clusters, function);
        }

        // Order the cluster by mean
        Collections.sort(clusters);

        return new KMeansClusteringAlgorithm(k, clusters);
    }

    @Override
    public double predict(Point point) {
        return Cluster.findClusterIndex(k, clusters(), point);
    }

    public List<Cluster> clusters() {
        return clusters;
    }

    public static class Cluster implements Comparable<Cluster> {
        private Point center;
        private final List<Point> points = new ArrayList<>();

        public Cluster(Point center) {
            this.center = center;
        }

        public static int findClusterIndex(int k, List<Cluster> clusters, Point point) {
            int bestClusterIndex = 0;
            double lowestDistance = -1;
            for (int i = 0; i < k; i++) {
                Cluster cluster = clusters.get(i);
                double xDiff = point.x() - cluster.center().x();
                double yDiff = point.y() - cluster.center().y();
                double distanceSquared = xDiff * xDiff + yDiff * yDiff;

                double dist = Math.sqrt(distanceSquared);
                if (lowestDistance == -1 || dist <= lowestDistance) {
                    bestClusterIndex = i;
                    lowestDistance = dist;
                }
            }
            return bestClusterIndex;
        }


        public static Cluster findCluster(int k, List<Cluster> clusters, Point point) {
            return clusters.get(findClusterIndex(k, clusters, point));
        }

        @Override
        public int compareTo(Cluster o) {
            double sum = 0;
            for (Point p : points()) {
                sum += p.x() + p.y();
            }

            double otherSum = 0;
            for (Point p : o.points()) {
                otherSum += p.x() + p.y();
            }

            double mean = sum / points().size();
            double otherMean = otherSum / o.points().size();
            return Double.compare(mean, otherMean);
        }

        @Override
        public Cluster clone() {
            List<Point> newPoints = new ArrayList<>();
            for (Point p : points()) {
                newPoints.add(p.clone());
            }
            Cluster cluster = new Cluster(center.clone());
            cluster.points().addAll(newPoints);
            return cluster;
        }

        public Point center() {
            return center;
        }

        public void center(Point center) {
            this.center = center;
        }

        public List<Point> points() {
            return points;
        }
    }
}
