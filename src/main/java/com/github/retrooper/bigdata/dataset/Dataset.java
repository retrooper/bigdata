package com.github.retrooper.bigdata.dataset;

import java.util.HashMap;
import java.util.Map;

public class Dataset<T, Z> {
    private final Map<T, Z> data = new HashMap<>();

    public Map<T, Z> getData() {
        return data;
    }

}
