package com.github.retrooper.bigdata.dataset;

import java.util.HashMap;
import java.util.Map;

public class Dataset<T extends Number, Z extends Number> {
    private final Map<T, Z> data = new HashMap<>();

    public Map<T, Z> getData() {
        return data;
    }

}
