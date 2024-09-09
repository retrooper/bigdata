package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.unsupervised.MarkovChainAlgorithm;
import com.github.retrooper.bigdata.algorithm.unsupervised.NGramAlgorithm;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

public class NGramConversationalMain {
    public static void main(String[] args) throws Exception {
        NGramAlgorithm nGram = new NGramAlgorithm(2);
        String[] fileNames = new String[] {"TwitterLowerAsciiCorpus.txt", "BNCCorpus.txt", "MovieCorpus.txt"};
        for (String fName : fileNames) {
           /* InputStream is = NGramConversationalMain.class.getResourceAsStream("/text_training/" + fName);
            StringBuilder trainingData = new StringBuilder();
            if (is != null) {
                try (InputStreamReader streamReader =
                             new InputStreamReader(is, StandardCharsets.UTF_8);
                     BufferedReader reader = new BufferedReader(streamReader)) {

                    String line;
                    while ((line = reader.readLine()) != null) {
                        trainingData.append(line);
                        //System.out.println("Erm");
                    }

                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            } else {
                System.out.println("Failed to find file!");
            }


            nGram.train(trainingData.toString());*/
        }

        nGram.train("Hey there! How are you? I am fine. Thank you. I missed you. I missed you too. That's always great to hear! Indeed my friend! Yep haha. Hello again! Howdy man! " +
                "Missed you a lot, really.. You mean it earnestly? Yes I do!");

        System.out.println("AI: I am ready when you are!...");
        Scanner scanner = new Scanner(System.in);
        while (true) {
            String line = scanner.nextLine();
            if (line.equals("QUIT")) {
                break;
            }
            line = line.replace("?", "").replace("!", "")
                            .replace(",", "");
            System.out.println("AI: " + nGram.generateText(line, 10));
        }
    }
}
