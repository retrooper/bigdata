package com.github.retrooper.bigdata.algorithm.unsupervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.dataset.Dataset;
import com.github.retrooper.bigdata.dataset.UnlabeledDatasetND;
import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class KMeansClusteringAlgorithm<Z extends NDimensionalPoint> implements LearningAlgorithm<Z> {
    private final int k;
    private final List<Cluster> clusters;

    private KMeansClusteringAlgorithm(int k, List<Cluster> clusters) {
        this.k = k;
        this.clusters = clusters;
    }

    private static void iteration(int k, List<Cluster> clusters, Dataset function) {
        function.iteratePoints(point -> {
            // Find cluster with center closest to point
            Cluster cluster = Cluster.findCluster(k, clusters, point);

            // Add the point to the best fitting cluster.
            cluster.points().add(point);
            return true;
        });


        for (Cluster cluster : clusters) {
            int n = cluster.points().size();
            int coordsLength = cluster.center.coordinates().length;
            if (coordsLength == 0) {
                coordsLength = cluster.points.get(0).coordinates().length;
            }
            float[] sums = new float[coordsLength];
            // Assume N coordinates
            for (int i = 0; i < n; i++) {
                NDimensionalPoint point = cluster.points().get(i);
                for (int j = 0; j < point.coordinates().length; j++) {
                    float coord = point.coordinates()[j];
                    sums[j] += coord;
                }
            }

            // Calculate means
            float[] means = new float[sums.length];
            for (int i = 0; i < means.length; i++) {
                means[i] = sums[i] / n;
            }

            cluster.center(new NDimensionalPoint(means));

            // New center is the mean of all points in that particular cluster
            //cluster.center(new Point(xSum / n, ySum / n));
        }
    }

    public static KMeansClusteringAlgorithm<NDimensionalPoint> build(int k, Dataset function, int iterations) {
        List<Cluster> clusters = new ArrayList<>(k);

        function.iteratePoints(point -> {
            clusters.add(new Cluster(point));
            // Condition to continue iterating
            return clusters.size() != k;
        });

        for (int i = 0; i < iterations; i++) {
            for (Cluster c : clusters) {
                c.points().clear();
            }
            iteration(k, clusters, function);
            System.out.println("K means interation at index: " + i);
        }

        // Order the cluster by mean
        Collections.sort(clusters);

        return new KMeansClusteringAlgorithm<>(k, clusters);
    }

    public static KMeansClusteringAlgorithm<NDimensionalPoint> build(int k, UnlabeledDatasetND function, int iterations) {
        List<Cluster> clusters = new ArrayList<>(k);

        function.iteratePoints(point -> {
            clusters.add(new Cluster(point));
            // Condition to continue iterating
            return clusters.size() != k;
        });

        for (int i = 0; i < iterations; i++) {
            for (Cluster c : clusters) {
                c.points().clear();
            }
            iteration(k, clusters, function);
            System.out.println("K means interation at index: " + i);
        }

        // Order the cluster by mean
        Collections.sort(clusters);

        return new KMeansClusteringAlgorithm<>(k, clusters);
    }


    @Override
    public float predict(Z point) {
        return Cluster.findClusterIndex(k, clusters(), point);
    }

    public List<Cluster> clusters() {
        return clusters;
    }

    public static class Cluster implements Comparable<Cluster> {
        private NDimensionalPoint center;
        private final List<NDimensionalPoint> points = new ArrayList<>();

        public Cluster(NDimensionalPoint center) {
            this.center = center;
        }

        public static int findClusterIndex(int k, List<Cluster> clusters, NDimensionalPoint point) {
            int bestClusterIndex = 0;
            float lowestDistance = Float.MAX_VALUE;
            for (int i = 0; i < k; i++) {
                Cluster cluster = clusters.get(i);
                float dist = point.distance(cluster.center);
                if (dist <= lowestDistance) {
                    bestClusterIndex = i;
                    lowestDistance = dist;
                }
            }
            return bestClusterIndex;
        }


        public static Cluster findCluster(int k, List<Cluster> clusters, NDimensionalPoint point) {
            return clusters.get(findClusterIndex(k, clusters, point));
        }

        @Override
        public int compareTo(Cluster o) {
            float sum = 0;
            for (NDimensionalPoint p : points()) {
                sum += p.getCoordinatesSum();
            }

            float otherSum = 0;
            for (NDimensionalPoint p : o.points()) {
                otherSum += p.getCoordinatesSum();
            }

            float mean = sum / points().size();
            float otherMean = otherSum / o.points().size();
            return Float.compare(mean, otherMean);
        }

        @Override
        public Cluster clone() {
            List<NDimensionalPoint> newPoints = new ArrayList<>();
            for (NDimensionalPoint p : points()) {
                newPoints.add(p.clone());
            }
            Cluster cluster = new Cluster(center.clone());
            cluster.points().addAll(newPoints);
            return cluster;
        }

        public NDimensionalPoint center() {
            return center;
        }

        public void center(NDimensionalPoint center) {
            this.center = center;
        }

        public List<NDimensionalPoint> points() {
            return points;
        }
    }
}
