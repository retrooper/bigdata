package com.github.retrooper.bigdata.text;

import java.util.Arrays;
import java.util.List;

public class Tokenizer {

    public List<String> tokenize(String input, boolean removeSymbols) {
        if (removeSymbols) {
            input = input.replace("?", "").replace(".", "").replace(",", "")
                    .replace("/", "").replace("!", "");
        }
        input = input.toLowerCase();
        return Arrays.stream(input.split(" ")).toList();
    }
}
