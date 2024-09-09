package com.github.retrooper.bigdata.algorithm.unsupervised;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class NGramAlgorithm {

    // Map to store n-gram frequencies: (n-1)-gram -> next word -> frequency
    private final Map<String, Map<String, Integer>> nGramMap = new HashMap<>();
    private final int n;  // n for n-gram model
    private final Random random = new Random();

    public NGramAlgorithm(int n) {
        this.n = n;
    }

    // Train the model on a text corpus
    public void train(String text) {
        String[] tokens = tokenize(text);
        for (int i = 0; i < tokens.length - n + 1; i++) {
            // Create the n-1 word sequence (the history)
            StringBuilder nGramKey = new StringBuilder();
            for (int j = 0; j < n - 1; j++) {
                if (j > 0) nGramKey.append(" ");
                nGramKey.append(tokens[i + j].toLowerCase());
            }

            String nextWord = tokens[i + n - 1].toLowerCase();  // The next word

            // Update the frequency map
            String nGramKeyStr = nGramKey.toString();
            nGramMap.putIfAbsent(nGramKeyStr, new HashMap<>());
            Map<String, Integer> nextWordFreq = nGramMap.get(nGramKeyStr);
            nextWordFreq.put(nextWord, nextWordFreq.getOrDefault(nextWord, 0) + 1);
        }
    }

    // Tokenize the input text (simple space-based tokenizer)
    public String[] tokenize(String text) {
        return text.split("\\s+");
    }

    // Generate a response starting from a seed of n-1 words
    public String generateText(String seed, int length) {
        StringBuilder generatedText = new StringBuilder(seed);
        String[] seedWords = tokenize(seed);
        String currentNGram = seed.toLowerCase();

        for (int i = 0; i < length; i++) {
            Map<String, Integer> possibleNextWords = nGramMap.getOrDefault(currentNGram.toLowerCase(), new HashMap<>());

            if (possibleNextWords.isEmpty()) {
                break;  // No possible next word, stop generating
            }

            String nextWord = getRandomNextWord(possibleNextWords);
            generatedText.append(" ").append(nextWord);

            // Update the current n-1 word sequence (sliding window)
            seedWords = tokenize(generatedText.toString());
            if (seedWords.length < n - 1) {
                break;  // Not enough words to continue the n-gram sequence
            }

            currentNGram = buildNGramKey(seedWords, seedWords.length - (n - 1), seedWords.length);
        }

        return generatedText.toString();
    }

    // Build the n-1 word sequence (key) from an array of words
    private String buildNGramKey(String[] words, int start, int end) {
        StringBuilder key = new StringBuilder();
        for (int i = start; i < end; i++) {
            if (i > start) key.append(" ");
            key.append(words[i].toLowerCase());
        }
        return key.toString();
    }

    // Get a random next word based on frequency
    private String getRandomNextWord(Map<String, Integer> nextWordFreq) {
        int totalFrequency = nextWordFreq.values().stream().mapToInt(Integer::intValue).sum();
        int randomIndex = random.nextInt(totalFrequency);

        int cumulativeFrequency = 0;
        for (Map.Entry<String, Integer> entry : nextWordFreq.entrySet()) {
            cumulativeFrequency += entry.getValue();
            if (cumulativeFrequency > randomIndex) {
                return entry.getKey();
            }
        }

        return null;  // Should never reach here
    }

}