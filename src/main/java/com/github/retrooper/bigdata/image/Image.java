package com.github.retrooper.bigdata.image;

import io.github.kamilszewc.opencv.OpenCV;
import io.github.kamilszewc.opencv.exception.SystemNotSupportedException;
import org.jetbrains.annotations.Nullable;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.function.Supplier;
public class Image {
    private static final float THRESHOLD = 200.0F;
    private final String path;
    private final Mat src;
    @Nullable
    private ImageFeatures features;
    public Image(String path) {
        try {
            OpenCV.loadLibrary();
        } catch (IOException | SystemNotSupportedException e) {
            throw new RuntimeException(e);
        }
        this.path = path;
        this.src = Imgcodecs.imread(path);
    }

    public String path() {
        return this.path;
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

            Core.convertScaleAbs(dstNorm, dstNormScaled);

            for( int i = 0; i < dstNorm.rows() ; i++ )
            {
                for( int j = 0; j < dstNorm.cols(); j++ )
                {
                    if (dstNorm.at(float.class, i, j).getV() > THRESHOLD) {
                        Imgproc.circle(dstNormScaled, new Point(j, i), 5, new Scalar(0), 2, 8, 0);
                    }
                }
            }

            // Don't save the processed image!
            //Imgcodecs.imwrite(path + "_PROCESSED" + ".jpg", dstNormScaled);

            float[] dstScaledData = new float[(int) (dstNorm.total() * dstNorm.channels())];
            dstNorm.get(0, 0, dstScaledData);

            double[] data = new double[dstScaledData.length];
            for (int i = 0; i < dstScaledData.length; i++) {
                data[i] = dstScaledData[i] / 100.0D;
            }

            return new ImageFeatures(data);
        };
    }
}
