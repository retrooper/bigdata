package com.github.retrooper.bigdata.image;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class Image {
    private final Mat src;

    private ImageFeatures features;
    public Image(String path) {
        this.src = Imgcodecs.imread(path);
    }

    public int width() {
        return src.width();
    }

    public int height() {
        return src.height();
    }
}
