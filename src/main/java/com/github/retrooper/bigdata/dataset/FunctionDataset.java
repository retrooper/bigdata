package com.github.retrooper.bigdata.dataset;

public class FunctionDataset<T, Z> extends Dataset<T, Z> {
    public FunctionDataset(T[] input, Z[] output) {
        for (int i = 0; i < input.length; i++) {
            getData().put(input[i], output[i]);
        }
    }

    public int dataPoints() {
        return getData().keySet().size();
    }
}
