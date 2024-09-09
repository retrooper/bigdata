package com.github.retrooper.bigdata;

import com.github.retrooper.bigdata.algorithm.supervised.SoftmaxRegressionAlgorithm;
import com.github.retrooper.bigdata.text.Tokenizer;
import com.github.retrooper.bigdata.text.mapper.SentenceVectorMapper;
import com.github.retrooper.bigdata.util.ClassificationHelper;
import com.github.retrooper.bigdata.util.NDimensionalPoint;

import java.util.Arrays;
import java.util.Scanner;

public class SoftmaxMain {
    public static void main(String[] args) {

        SentenceVectorMapper sentence = new SentenceVectorMapper(50);
        Tokenizer tokenizer = new Tokenizer();
        float[][] trainingData = {
                //Class 0 - Asks of the AI's emotional status
                sentence.sentenceVector(Arrays.asList("hey", "how", "are", "you", "doing", "buddy")),
                sentence.sentenceVector(Arrays.asList("how's", "it", "going")),
                sentence.sentenceVector(Arrays.asList("how", "are", "you")),
                sentence.sentenceVector(Arrays.asList("you", "doing", "alright")),
                sentence.sentenceVector(Arrays.asList("you", "doing", "fine")),
                sentence.sentenceVector(Arrays.asList("you", "fine")),

                //Class 1 - Positive response about emotional status
                sentence.sentenceVector(Arrays.asList("hey", "i'm", "doing", "great", "thanks", "for", "asking")),
                sentence.sentenceVector(Arrays.asList("hey", "i'm", "doing", "fine", "thanks")),
                sentence.sentenceVector(Arrays.asList("hey", "i'm", "okay")),
                sentence.sentenceVector(Arrays.asList("i'm", "okay")),
                sentence.sentenceVector(Arrays.asList("i'm", "doing", "good")),
                sentence.sentenceVector(Arrays.asList("good", "and", "you")),

                //Class 2 - Negative response about emotional status
                sentence.sentenceVector(Arrays.asList("hey", "i'm", "doing", "bad", "thanks")),
                sentence.sentenceVector(Arrays.asList("hey", "i'm", "doing", "bad")),
                sentence.sentenceVector(Arrays.asList("hey", "i'm", "not", "great")),
                sentence.sentenceVector(Arrays.asList("i'm", "sad")),
                sentence.sentenceVector(Arrays.asList("horrible")),
                sentence.sentenceVector(Arrays.asList("terrible")),

                //Class 3 - Requesting identification of the AI
                sentence.sentenceVector(Arrays.asList("what", "is", "your", "name")),
                sentence.sentenceVector(Arrays.asList("hey", "what", "is", "your", "name")),
                sentence.sentenceVector(Arrays.asList("who", "are", "you")),
                sentence.sentenceVector(Arrays.asList("you", "are")),
                sentence.sentenceVector(Arrays.asList("and", "you", "are")),
                sentence.sentenceVector(Arrays.asList("what", "are", "you")),

                //Class 4 - Greetings
                sentence.sentenceVector(Arrays.asList("hey")),
                sentence.sentenceVector(Arrays.asList("hi")),
                sentence.sentenceVector(Arrays.asList("heyo")),
                sentence.sentenceVector(Arrays.asList("sup")),
                sentence.sentenceVector(Arrays.asList("wussup")),
                sentence.sentenceVector(Arrays.asList("salutations")),

                //Class 5 - Request for help
                sentence.sentenceVector(Arrays.asList("i", "need", "help")),
                sentence.sentenceVector(Arrays.asList("i", "need", "your", "help")),
                sentence.sentenceVector(Arrays.asList("can", "you", "help", "me")),
                sentence.sentenceVector(Arrays.asList("could", "you", "help", "me")),
                sentence.sentenceVector(Arrays.asList("i'm", "stuck")),
                sentence.sentenceVector(Arrays.asList("please", "help", "me")),
        };

        int[] classification = new int[] {
                0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4,
                5, 5, 5, 5, 5, 5,
        };


        SoftmaxRegressionAlgorithm<NDimensionalPoint> algorithm =
                new SoftmaxRegressionAlgorithm<>(ClassificationHelper.countDistinct(classification),
                        0.1F, 1000);

        algorithm.fit(trainingData, classification);

        System.out.println("I'm ready when you are!");
        Scanner scanner = new Scanner(System.in);
        while (true) {
            String line = scanner.nextLine();
            if (line.equals("QUIT")) break;
            int prediction = (int) algorithm.predict(new NDimensionalPoint(
                    sentence.sentenceVector(tokenizer.tokenize(line, true))));
            if (prediction == 0) {
                System.out.println("I hope this message finds you well. I'm doing good, dear user!");
            }
            else if (prediction == 1) {
                System.out.println("Great! That's really good to hear.");
            }
            else if (prediction == 2) {
                System.out.println("Oh no... I'm sorry to hear that. I revere anyone that is able to divulge.");
            }
            else if (prediction == 3) {
                System.out.println("I am the HowAreYou virtual assistant.");
            }
            else if (prediction == 4) {
                System.out.println("Greetings! It's always great to hear from you.");
            }
            else if (prediction == 5) {
                System.out.println("Sure thing! Don't worry about it. That's what I'm here for, right? What do you need?");

            }
        }
    }
}
