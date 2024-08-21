package com.github.retrooper.bigdata.image;

import io.github.kamilszewc.opencv.OpenCV;
import io.github.kamilszewc.opencv.exception.SystemNotSupportedException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import java.io.IOException;
import java.util.function.Supplier;
public class Image {
    private static final float THRESHOLD = 200.0F;
    private final String path;
    private Mat src;
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

    public void resize(int width, int height) {
        Mat dst = new Mat();
        Imgproc.resize(src, dst, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR_EXACT);
        src = dst;
    }

    public void save() {
        Imgcodecs.imwrite(path, src);
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

            HOGDescriptor descriptor = new HOGDescriptor();
            MatOfFloat features = new MatOfFloat();
            descriptor.compute(srcGray, features);
            /*
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
            }*/



            // Don't save the processed image!
            //Imgcodecs.imwrite(path + "_PROCESSED" + ".jpg", features.t());
            return new ImageFeatures(features.toArray());
        };
    }
}
