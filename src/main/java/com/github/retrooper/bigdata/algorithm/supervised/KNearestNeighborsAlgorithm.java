package com.github.retrooper.bigdata.algorithm.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.dataset.SimpleLabeledDatasetND;
import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.util.*;

public class KNearestNeighborsAlgorithm<Z extends NDimensionalPoint> implements LearningAlgorithm<Z> {
    private final int k;
    private final List<DataPoint> points;

    protected KNearestNeighborsAlgorithm(int k, List<DataPoint> points) {
        this.k = k;
        this.points = points;

    }

    public static KNearestNeighborsAlgorithm<NDimensionalPoint> build(int k, SimpleLabeledDatasetND function) {
        List<DataPoint> points = new ArrayList<>();
        function.iterate((point, value) -> {
            points.add(new DataPoint(point, value));
            return true;
        });
        return new KNearestNeighborsAlgorithm<>(k, points);
    }

    @Override
    public float predict(Z x) {
        points.sort((a, b) -> Float.compare(a.point.distance(x), b.point.distance(x)));

        int freqA = 0;
        int freqB = 0;
        for (int i = 0; i < k; i++) {
            if (points.get(i).clusterValue == 0) {
                freqA++;
            }
            else if (points.get(i).clusterValue == 1) {
                freqB++;
            }
        }

        return (freqA > freqB) ?  0 : 1;
    }

    public int k() {
        return k;
    }

    public static class DataPoint {
        private final NDimensionalPoint point;
        private final int clusterValue;

        public DataPoint(NDimensionalPoint point, int clusterValue) {
            this.point = point;
            this.clusterValue = clusterValue;
        }

        public NDimensionalPoint getPoint() {
            return point;
        }

        public int getClusterValue() {
            return clusterValue;
        }
    }
}
