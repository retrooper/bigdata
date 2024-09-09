package com.github.retrooper.bigdata.util;

public class ClassificationHelper {
    public static int countDistinct(int[] arr) {
        int res = 1;

        for (int i = 1; i < arr.length; i++) {
            int j = 0;
            for (j = 0; j < i; j++)
                if (arr[i] == arr[j])
                    break;
            if (i == j)
                res++;
        }
        return res;
    }

}
