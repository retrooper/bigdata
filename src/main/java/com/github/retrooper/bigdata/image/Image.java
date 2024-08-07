package com.github.retrooper.bigdata.image;

import io.github.kamilszewc.opencv.OpenCV;
import io.github.kamilszewc.opencv.exception.SystemNotSupportedException;
import org.jetbrains.annotations.Nullable;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.function.Supplier;

public class Image {
    private final Mat src;
    @Nullable
    private ImageFeatures features;
    public Image(String path) {
        try {
            OpenCV.loadLibrary();
        } catch (IOException | SystemNotSupportedException e) {
            throw new RuntimeException(e);
        }
        this.src = Imgcodecs.imread(path);
    }

    public int width() {
        return src.width();
    }

    public int height() {
        return src.height();
    }

    public Supplier<ImageFeatures> features() {
        return () -> {
            try {
                OpenCV.loadLibrary();
            } catch (IOException | SystemNotSupportedException e) {
                throw new RuntimeException(e);
            }
            Mat srcGray = new Mat();
            Imgproc.cvtColor(src, srcGray, Imgproc.COLOR_BGR2GRAY);
            Mat dst = new Mat();
            Mat dstNorm = new Mat();
            Mat dstNormScaled = new Mat();

            int blockSize = 2;
            int apertureSize = 3;
            double k = 0.04;

            Imgproc.cornerHarris(srcGray, dst, blockSize, apertureSize, k);

            /// Normalizing
            Core.normalize(dst, dstNorm, 0, 255, Core.NORM_MINMAX);

            float[] dstNormData = new float[(int) (dstNorm.total() * dstNorm.channels())];
            dstNorm.get(0, 0, dstNormData);
            Core.convertScaleAbs(dstNorm, dstNormScaled);

            double[] data = new double[dstNormData.length];
            for (int i = 0; i < dstNormData.length; i++) {
                data[i] = dstNormData[i];
            }

            return new ImageFeatures(data);
        };
    }
}
