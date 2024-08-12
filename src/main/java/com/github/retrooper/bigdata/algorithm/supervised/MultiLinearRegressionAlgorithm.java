package com.github.retrooper.bigdata.algorithm.supervised;

import com.github.retrooper.bigdata.algorithm.LearningAlgorithm;
import com.github.retrooper.bigdata.dataset.LabeledDatasetND;
import com.github.retrooper.bigdata.dataset.SimpleLabeledDatasetND;
import com.github.retrooper.bigdata.util.NDimensionalPoint;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Map;
import java.util.function.Predicate;

public class MultiLinearRegressionAlgorithm implements LearningAlgorithm<Float[]> {
    private final float learningRate = 0.01F;
    private float[] inputs;
    private float beta;
    private float[] weights;
    private int epochs;

    private MultiLinearRegressionAlgorithm(float[] inputs, float beta, float[] weights, int epochs) {

    }

    @Override
    public float predict(Float[] x) {
        float result = 0.0F;
        for (int i = 0; i < inputs.length; i++) {
            result = inputs[i] * weights[i] + result;
        }
        return result + beta;
    }

    public void train(LabeledDatasetND function, float[] result, int features, int epochs) {
        for (int e = 0; e < epochs; e++) {
            float mse = 0.0F;
            int i = 0;
            for (Map.Entry<Float[], Float[]> entry : function.getData().entrySet()) {
                Float[] tempInputs = entry.getKey();
                MultiLinearRegressionAlgorithm mlr = new MultiLinearRegressionAlgorithm(ArrayUtils.toPrimitive(tempInputs),
                        beta, weights, epochs);
                float value = mlr.predict(tempInputs);
                float error = value - result[i];
                mse = error * error + mse;

                for (int j = 0; j < weights.length; j++) {
                    weights[j] = weights[j] - learningRate * error * tempInputs[j];
                }
                beta = beta - learningRate * error;

                i++;
            }
            mse = (float) ((Math.sqrt(mse)) / function.dataPoints());
        }

    }
}
}
