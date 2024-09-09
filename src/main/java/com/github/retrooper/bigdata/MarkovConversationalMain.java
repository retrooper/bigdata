package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.unsupervised.MarkovChainAlgorithm;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import java.util.Scanner;

public class MarkovConversationalMain {
    public static void main(String[] args) throws Exception {
        MarkovChainAlgorithm markov = new MarkovChainAlgorithm();
        String[] fileNames = new String[] {"TwitterLowerAsciiCorpus.txt", "BNCCorpus.txt", "MovieCorpus.txt"};
        for (String fName : fileNames) {
            InputStream is = MarkovConversationalMain.class.getResourceAsStream("/text_training/" + fName);
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


            markov.train(trainingData.toString());
        }

        System.out.println("AI: I am ready when you are!...");
        Scanner scanner = new Scanner(System.in);
        while (true) {
            String line = scanner.nextLine();
            if (line.equals("QUIT")) {
                break;
            }
            line = line.replace("?", "").replace("!", "")
                            .replace(",", "");
            System.out.println("AI: " + markov.generateText(line, 6));
        }
    }
}
