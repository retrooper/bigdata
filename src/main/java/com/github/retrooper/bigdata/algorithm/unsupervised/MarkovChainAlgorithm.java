package com.github.retrooper.bigdata.algorithm.unsupervised;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class MarkovChainAlgorithm {
    private final Map<String, Map<String, Integer>> wordMap = new HashMap<>();
    private final Random random = new Random();

    public void train(String text) {
        text = text.replace("!", "").replace("?", "").replace(",", "").replace(".", "");
        //Tokenize
        String[] words = text.split(" ");
        for (int i = 0; i < words.length; i++) {
            String cw = words[i].toLowerCase();
            if (i + 1 >= words.length) {
                break;
            }
            String nw = words[i + 1].toLowerCase();

            wordMap.put(cw, new HashMap<>());
            wordMap.get(cw).put(nw, wordMap.get(cw).getOrDefault(nw, 0) + 1);
        }
    }

    public String generateText(String seed, int len) {
        StringBuilder response = new StringBuilder(seed);
        String currentWord = seed.toLowerCase();

        for (int i = 0; i < len; i++) {
            Map<String, Integer> nextWords = wordMap.getOrDefault(currentWord, new HashMap<>());
            if (nextWords.isEmpty()) {
                boolean found = false;
                //They still want responses, find something
                for (String key : wordMap.keySet()) {
                    if (response.toString().toLowerCase().contains(key) && !key.equals(" ")) {
                        nextWords.put(key + " ", 1);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // No next words, end response
                    break;
                }
            }

            // Choose the next word based on probabilities
            currentWord = getRandomNextWord(nextWords);
            response.append(" ").append(currentWord);
        }

        String text = response.toString();
        StringBuilder newText = new StringBuilder();
        for (char c : text.toCharArray()) {
            newText.append(c);
            if (c == '.' || c == '!' || c == '?') {
                break;
            }
        }
        return newText.toString();
    }

    private String getRandomNextWord(Map<String, Integer> nextWords) {
        int totalFrequency = nextWords.values().stream().mapToInt(Integer::intValue).sum();
        int randomIndex = random.nextInt(totalFrequency);

        int cumulativeFrequency = 0;
        for (Map.Entry<String, Integer> entry : nextWords.entrySet()) {
            cumulativeFrequency += entry.getValue();
            if (cumulativeFrequency > randomIndex) {
                return entry.getKey();
            }
        }

        return null; // Shouldn't get here
    }


}
