package com.github.retrooper.bigdata.algorithm.unsupervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.dataset.FunctionDataset2D;
import com.github.retrooper.bigdata.dataset.FunctionDatasetNDimensional;
import com.github.retrooper.bigdata.util.NDimensionalPoint;
import com.github.retrooper.bigdata.util.Point;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class KMeansClusteringAlgorithm<Z extends NDimensionalPoint<Double>> implements LearningAlgorithm<Z> {
    private final int k;
    private final List<Cluster> clusters;

    private KMeansClusteringAlgorithm(int k, List<Cluster> clusters) {
        this.k = k;
        this.clusters = clusters;
    }

    private static void iteration(int k, List<Cluster> clusters, FunctionDataset2D<Double, Double> function) {
        function.iteratePoints(point -> {
            // Find cluster with center closest to point
            Cluster cluster = Cluster.findCluster(k, clusters, point);

            // Add the point to the best fitting cluster.
            cluster.points().add(point);
            return true;
        });

        for (Cluster cluster : clusters) {
            int n = cluster.points().size();
            List<Double> coordSum = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                NDimensionalPoint<Double> point = cluster.points().get(i);
                for (double coord : point.coordinates()) {

                }
                //xSum += point.x();
                //ySum += point.y();
            }

            // New center is the mean of all points in that particular cluster
            //cluster.center(new Point(xSum / n, ySum / n));
        }
    }

    public static KMeansClusteringAlgorithm build(int k, FunctionDataset2D<Double, Double> function) {
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

    public static KMeansClusteringAlgorithm build(int k, FunctionDatasetNDimensional<Double, Double> function) {
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
            //iteration(k, clusters, function);
        }

        // Order the cluster by mean
        Collections.sort(clusters);

        return new KMeansClusteringAlgorithm(k, clusters);
    }


    @Override
    public double predict(Z point) {
        return Cluster.findClusterIndex(k, clusters(), point);
    }

    public List<Cluster> clusters() {
        return clusters;
    }

    public static class Cluster implements Comparable<Cluster> {
        private NDimensionalPoint<Double> center;
        private final List<NDimensionalPoint<Double>> points = new ArrayList<>();

        public Cluster(NDimensionalPoint<Double> center) {
            this.center = center;
        }

        public static int findClusterIndex(int k, List<Cluster> clusters, NDimensionalPoint<Double> point) {
            int bestClusterIndex = 0;
            double lowestDistance = -1;
            for (int i = 0; i < k; i++) {
                Cluster cluster = clusters.get(i);
                double dist = point.distance(cluster.center);
                if (lowestDistance == -1 || dist <= lowestDistance) {
                    bestClusterIndex = i;
                    lowestDistance = dist;
                }
            }
            return bestClusterIndex;
        }


        public static Cluster findCluster(int k, List<Cluster> clusters, NDimensionalPoint<Double> point) {
            return clusters.get(findClusterIndex(k, clusters, point));
        }

        @Override
        public int compareTo(Cluster o) {
            double sum = 0;
            for (NDimensionalPoint<Double> p : points()) {
                sum += p.getCoordinatesSum();
            }

            double otherSum = 0;
            for (NDimensionalPoint<Double> p : o.points()) {
                otherSum += p.getCoordinatesSum();
            }

            double mean = sum / points().size();
            double otherMean = otherSum / o.points().size();
            return Double.compare(mean, otherMean);
        }

        @Override
        public Cluster clone() {
            List<NDimensionalPoint<Double>> newPoints = new ArrayList<>();
            for (NDimensionalPoint<Double> p : points()) {
                newPoints.add(p.clone());
            }
            Cluster cluster = new Cluster(center.clone());
            cluster.points().addAll(newPoints);
            return cluster;
        }

        public NDimensionalPoint<Double> center() {
            return center;
        }

        public void center(NDimensionalPoint<Double> center) {
            this.center = center;
        }

        public List<NDimensionalPoint<Double>> points() {
            return points;
        }
    }
}
