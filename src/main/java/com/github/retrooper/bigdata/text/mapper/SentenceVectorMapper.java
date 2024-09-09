package com.github.retrooper.bigdata.text.mapper;

import java.util.List;

public class SentenceVectorMapper {
    private final int vectorSize;
    private final SimpleWordVectorMapper simpleWordVectorMapper;

    public SentenceVectorMapper(int vectorSize) {
        this.vectorSize = vectorSize;
        this.simpleWordVectorMapper = new SimpleWordVectorMapper(vectorSize);
    }

    public float[] sentenceVector(List<String> words) {
        float[] averageForSentence = new float[vectorSize];

        //Calculate average for words
        for (String word : words) {
            float[] wordVector = simpleWordVectorMapper.getWordAsVector(word);
            for (int i = 0; i < vectorSize; i++) {
                averageForSentence[i] += wordVector[i];
            }
        }

        //Divide by N to obtain average
        for (int i = 0; i < vectorSize; i++) {
            averageForSentence[i] /= words.size();
        }

        return averageForSentence;
    }
}
