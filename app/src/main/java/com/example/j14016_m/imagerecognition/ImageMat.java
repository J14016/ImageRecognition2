package com.example.j14016_m.imagerecognition;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.CvType.CV_8UC4;

/**
 * Created by J14016_M on 2018/01/09.
 */

public class ImageMat {
    Mat img;
    private static Resources mResources;

    public ImageMat(int id) {
        Bitmap src = BitmapFactory.decodeResource(mResources, id);
        img = new Mat(src.getHeight(), src.getWidth(), CV_8UC4);
        Utils.bitmapToMat(src, img);
    }

    public ImageMat(int id, int type) {
        Bitmap src = BitmapFactory.decodeResource(mResources, id);
        img = new Mat(src.getHeight(), src.getWidth(), type);
        Utils.bitmapToMat(src, img);
    }

    public static Mat loadImage(int id) {
        Mat result;
        Bitmap src = BitmapFactory.decodeResource(mResources, id);
        result = new Mat(src.getHeight(), src.getWidth(), CV_8UC4);
        Utils.bitmapToMat(src, result);

        return result;
    }

    public static void setResources(Resources resources) {
        mResources = resources;
    }

    public static Resources getResources() {
        return mResources;
    }

    public static Bitmap toBitmap(Mat mat) {
        Bitmap bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        return bitmap;
    }

    public static Mat toMat(Bitmap bitmap) {
        Mat img = new Mat(bitmap.getHeight(), bitmap.getWidth(), CV_8UC4);
        Utils.bitmapToMat(bitmap, img);
        return img;
    }

    public static void drawMat(Canvas canvas, Mat mat, int x, int y, int width, int height) {
        Bitmap bitmap = toBitmap(mat);

        if(width == 0) {
            width = bitmap.getWidth();
        }
        if(height == 0) {
            height = bitmap.getHeight();
        }

        canvas.drawBitmap(bitmap, new android.graphics.Rect(0, 0, bitmap.getWidth(), bitmap.getHeight()), new android.graphics.Rect(x, y, x + width, y + height), new Paint());
    }

    public static Mat toGrayscale(Mat mat) {
        Mat gray = new Mat();
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGB2GRAY);
        return gray;
    }
}
