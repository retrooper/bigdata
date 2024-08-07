package com.github.retrooper.bigdata.dataset;

import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class FunctionDatasetNDimensional implements Dataset {
    private final Map<Double[], Double[]> data = new HashMap<>();
    public FunctionDatasetNDimensional(Double[][] input, Double[][] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public FunctionDatasetNDimensional(double[][] input, double[][] output) {
        Double[][] inputBoxed = new Double[input.length][];
        for (int i = 0; i < input.length; i++) {
            inputBoxed[i] = Arrays.stream(input[i]).boxed().toArray(Double[]::new);
        }

        Double[][] outputBoxed = new Double[output.length][];
        for (int i = 0; i < output.length; i++) {
            outputBoxed[i] = Arrays.stream(output[i]).boxed().toArray(Double[]::new);
        }

        for (int i = 0; i < input.length; i++) {
            getData().put(inputBoxed[i], outputBoxed[i]);
        }
    }

    public FunctionDatasetNDimensional(Double[][] input) {
        for (Double[] doubles : input) {
            getData().put(doubles, null);
        }
    }


    // Memory intensive
    @Deprecated
    public FunctionDatasetNDimensional(double[][] input) {
        for (double[] array : input) {
            Double[] boxed = Arrays.stream(array).boxed().toArray(Double[]::new);
            getData().put(boxed, null);
        }
    }


    @Override
    public int dataPoints() {
        return getData().keySet().size();
    }

    @Override
    public void iteratePoints(Predicate<NDimensionalPoint> consumer) {
        for (Map.Entry<Double[], Double[]> entry : getData().entrySet()) {

            NDimensionalPoint point = new NDimensionalPoint(entry.getKey());
            if (!consumer.test(point)) break;
        }
    }

    public Map<Double[], Double[]> getData() {
        return data;
    }
}