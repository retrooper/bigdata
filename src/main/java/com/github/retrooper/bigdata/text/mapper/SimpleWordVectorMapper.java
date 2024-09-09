package com.github.retrooper.bigdata.text.mapper;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class SimpleWordVectorMapper {
    private final int vectorSize;
    private final Map<String, float[]> vectorMap = new HashMap<>();
    private final Random random = new Random();
    public SimpleWordVectorMapper(int vectorSize) {
        this.vectorSize= vectorSize;
    }


    public float[] getWordAsVector(String word) {
        return vectorMap.computeIfAbsent(word, k -> randomVector());
    }

    private float[] randomVector() {
        float[] vector = new float[vectorSize];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = random.nextFloat() - 0.5F;
        }
        return vector;
    }
}
