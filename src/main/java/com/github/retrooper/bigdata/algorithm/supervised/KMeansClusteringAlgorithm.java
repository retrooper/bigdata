package com.github.retrooper.bigdata.algorithm.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset;
import com.github.retrooper.bigdata.util.Point;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

public class KMeansClusteringAlgorithm implements LearningAlgorithm<Point> {
    private final int k;
    private final List<Cluster> clusters;

    private KMeansClusteringAlgorithm(int k, List<Cluster> clusters) {
        this.k = k;
        this.clusters = clusters;
    }

    private static void iteration(int k, List<Cluster> clusters, FunctionDataset<Double, Double> function) {
        function.iteratePoints(new Predicate<Point>() {
            @Override
            public boolean test(Point point) {
                // Find cluster with center closest to point
                Cluster cluster = Cluster.findCluster(k, clusters, point);

                // Add the point to the best fitting cluster.
                cluster.points().add(point);
                return true;
            }
        });

        // Iteration 2

        for (Cluster cluster : clusters) {
            int n = cluster.points().size();
            double xSum = 0;
            double ySum = 0;
            for (int i = 0; i < n; i++) {
                Point point = cluster.points().get(i);
                xSum = point.x();
                ySum = point.y();
            }

            // New center is the mean of all points in that particular cluster
            cluster.center(new Point(xSum / n, ySum / n));
        }
    }

    public static KMeansClusteringAlgorithm build(int k, FunctionDataset<Double, Double> function) {
        List<Cluster> clusters = new ArrayList<>(k);

        function.iteratePoints(new Predicate<Point>() {
            @Override
            public boolean test(Point point) {
                clusters.add(new Cluster(point));
                // Condition to continue iterating
                return clusters.size() != k;
            }
        });


        iteration(k, clusters, function);
        iteration(k, clusters, function);

        return new KMeansClusteringAlgorithm(k, clusters);
    }

    @Override
    public double predict(Point point) {
        return Cluster.findClusterIndex(k, clusters(), point);
    }

    public List<Cluster> clusters() {
        return clusters;
    }

    public static class Cluster {
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
