package com.example.j14016_m.imagerecognition;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.AKAZE;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY;
import static org.opencv.imgproc.Imgproc.GaussianBlur;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.getDefaultNewCameraMatrix;
import static org.opencv.imgproc.Imgproc.polylines;

/**
 * Created by J14016_M on 2018/01/15.
 */

public class ImageDetectFragment extends Fragment implements Runnable {

    private final static String TAG = "ImageDetect";

    private Thread mThread = null;
    private volatile boolean running = false;
    private TextureView mTextureView;
    private Bitmap mCameraImage;

    private static final String ARGS_NAME = "camera_image";

    private FeatureDetector mDetector;
    private DescriptorExtractor mExtractor;
    private DescriptorMatcher mMatcher;

    private AKAZE akaze = AKAZE.create();

    private List<Mat> mTemplateImages = new ArrayList<>();
    private List<Mat> mGrayTemplateImages = new ArrayList<>();
    private List<Mat> mTrainDescriptorses = new ArrayList<>();

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_image_detect, container, false);
    }

    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        mTextureView = view.findViewById(R.id.image_detect_texture);
        mTextureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {

            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {

            }

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                return false;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surface) {

            }
        });

        Button cameraButton = (Button)view.findViewById(R.id.camera);
        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                switch(v.getId()) {
                    case R.id.camera: {
                        FragmentManager fragmentManager = getActivity().getSupportFragmentManager();
                        fragmentManager.beginTransaction().replace(R.id.container, Camera2BasicFragment.newInstance()).commit();
                        break;
                    }
                }
            }
        });

        //Camera2BasicFragment camera2BasicFragment = (Camera2BasicFragment)getParentFragment();

    }

    @Override
    public void run() {

        Mat cameraImageMat = ImageMat.toMat(mCameraImage);

        //Mat grayCameraImage = ImageMat.toGrayscale(cameraImageMat);
        Mat grayCameraImage = new Mat();
        Imgproc.cvtColor(cameraImageMat, grayCameraImage, COLOR_RGB2GRAY);

        Imgproc.threshold(grayCameraImage, grayCameraImage, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(grayCameraImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Mat> matList = new ArrayList<>();

        for(int i = 0; i < contours.size(); i++) {
            MatOfPoint2f contours2f = new MatOfPoint2f(contours.get(i).toArray());
            MatOfPoint2f approx2f = new MatOfPoint2f();
            Imgproc.approxPolyDP(contours2f, approx2f, 0.1 * Imgproc.arcLength(contours2f, true), true);

            MatOfPoint approx = new MatOfPoint(approx2f.toArray());
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(approx, hull);

            if(hull.size().height == 4) {
                double area = Imgproc.contourArea(approx);
                if(area > 1000) {
                    double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                    ArrayList<Integer> srcPoints = new ArrayList<>();
                    for(int k = 0; k < hull.size().height; k++) {
                        int hullIndex = (int)hull.get(k, 0)[0];
                        double[] m = approx.get(hullIndex, 0);
                        //srcPoints.add((int)m[0]);
                        //srcPoints.add((int)m[1]);

                        if(k == 0) {
                            x1 = m[0];
                            y1 = m[1];
                            x2 = m[0];
                            y2 = m[1];
                        } else {
                            if(x1 > m[0]) x1 = m[0];
                            if(y1 > m[1]) y1 = m[1];
                            if(x2 < m[0]) x2 = m[0];
                            if(y2 < m[1]) y2 = m[1];
                        }
                    }

                    matList.add(new Mat(grayCameraImage, new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1))));
                }
            }
        }

        contours = new ArrayList<>();
        Mat outImage = mGrayTemplateImages.get(0);
        if(matList.size() > 0) {
            Imgproc.findContours(outImage, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            for(int i = 0; i < contours.size(); i++) {
                MatOfPoint2f contours2f = new MatOfPoint2f(contours.get(i).toArray());
                MatOfPoint2f approx2f = new MatOfPoint2f();
                Imgproc.approxPolyDP(contours2f, approx2f, 2 * Imgproc.arcLength(contours2f, true), true);

                MatOfPoint approx = new MatOfPoint(approx2f.toArray());
                MatOfInt hull = new MatOfInt();
                Imgproc.convexHull(approx, hull);

                double area = Imgproc.contourArea(approx);
                if(area > 1000) {
                    //Imgproc.polylines(outImage, approx, true, new Scalar(255, 0, 0, 255));
                    List<MatOfPoint> pts = new ArrayList<>();
                    pts.add(approx);
                    outImage = mTemplateImages.get(0);
                    Imgproc.polylines(outImage, pts, true, new Scalar(255, 0, 0, 255));

                }

            }
        }



        while(running) {
            Canvas canvas = mTextureView.lockCanvas();

            if(canvas != null) {
                canvas.drawColor(Color.GRAY);
                if(mCameraImage != null) {

                    if(matList.size() > 0) {
                        ImageMat.drawMat(canvas, outImage, 0, 0, 0, 0);
                        //ImageMat.drawMat(canvas, mGrayTemplateImages.get(0), matList.get(0).width(), 0, 0, 0);
                        //ImageMat.drawMat(canvas, featureMatching(matList.get(i), mGrayTemplateImages.get(0)), 0, 0, 100, 100);
                        featureMatching(canvas, matList.get(0));
                    }
                }
            }
            mTextureView.unlockCanvasAndPost(canvas);

        }
    }

    public static ImageDetectFragment newInstance() {
        ImageDetectFragment fragment = new ImageDetectFragment();
        return fragment;
    }


    public Mat featureMatching(Canvas canvas, Mat imgGrayMat) {

        MatOfKeyPoint queryKeypoints = new MatOfKeyPoint();
        Mat queryDes = new Mat();


        /*
        mDetector.detect(imgGrayMat, queryKeypoints);
        mExtractor.compute(imgGrayMat, queryKeypoints, queryDes);
        */
        akaze.detect(imgGrayMat, queryKeypoints);
        akaze.compute(imgGrayMat, queryKeypoints, queryDes);

        List<MatOfDMatch> matches = new ArrayList<>();
        mMatcher.knnMatch(queryDes, matches, 2);

        double threshold = .7;


        List<DMatch> goodMatches = new ArrayList<>();
        for(MatOfDMatch match : matches) {
            double dist1 = match.toArray()[0].distance;
            double dist2 = match.toArray()[1].distance;

            if(dist1 <= dist2 * threshold) {
                goodMatches.add(match.toArray()[0]);
            }
        }

        Collections.sort(goodMatches, new Comparator<DMatch>() {
            @Override
            public int compare(DMatch lhs, DMatch rhs) {
                if(lhs.distance < rhs.distance) {
                    return -1;
                } else if(lhs.distance == rhs.distance) {
                    return 0;
                } else if(lhs.distance > rhs.distance) {
                    return 1;
                }

                return 0;
            }

        });

        /*
        for(DMatch dMatch : goodMatches) {
            Log.d("distance", "" + dMatch.distance + ", " + dMatch.imgIdx);
        }
        */

        for(int i = 0; i < goodMatches.size(); i++) {
            if (goodMatches.size() > 0)
                Log.d("distance", goodMatches.get(i).distance + ", " + goodMatches.get(i).imgIdx);
        }
        /*
        List<Point> pts1 = new ArrayList<>();
        List<Point> pts2 = new ArrayList<>();
        for(DMatch dMatch : goodMatches) {
            pts1.add(queryKeypoints.toList().get(dMatch.queryIdx).pt);
            pts2.add(trainKeypoints.toList().get(dMatch.trainIdx).pt);
        }

        Mat outputMask = new Mat();
        MatOfPoint2f pts1Mat = new MatOfPoint2f();
        pts1Mat.fromList(pts1);
        MatOfPoint2f pts2Mat = new MatOfPoint2f();
        pts2Mat.fromList(pts2);

        Mat Homog = Calib3d.findHomography(pts1Mat, pts2Mat, Calib3d.RANSAC, 15);

        List<DMatch> betterMatches = new ArrayList<>();
        for(int i = 0; i < goodMatches.size(); i++) {
            if(outputMask.get(i, 0)[0] != 0.0) {
                betterMatches.add(goodMatches.get(i));
            }
        }



        Mat outputImg = new Mat();
        // this will draw all matches, works fine
        MatOfDMatch better_matches_mat = new MatOfDMatch();
        better_matches_mat.fromList(betterMatches);
        Features2d.drawMatches(imgGrayMat, queryKeypoints, tmpMat, trainKeypoints, better_matches_mat, outputImg);
        */
        Mat outputImg = new Mat();

        return outputImg;

    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        Bundle bundle = getArguments();
        /*
        Bitmap bitmap = BitmapFactory.decodeByteArray(bundle.getByteArray("camera_image"), 0, bundle.getInt("camera_image_length"));
        */
        mCameraImage = ((AutoFitTextureView)bundle.getSerializable("camera_image")).getBitmap();


        mTemplateImages.add(ImageMat.loadImage(R.drawable.iwan));
        mTemplateImages.add(ImageMat.loadImage(R.drawable.ryanwan));
        mTemplateImages.add(ImageMat.loadImage(R.drawable.isou));
        mTemplateImages.add(ImageMat.loadImage(R.drawable.ipin));


        for(Mat tmpImage : mTemplateImages) {
            Mat grayImage = new Mat();
            Imgproc.cvtColor(tmpImage, grayImage, COLOR_RGBA2GRAY);
            //GaussianBlur(tmpImage, tmpImage, new Size(5, 5), 0);
            Imgproc.threshold(grayImage, grayImage, 127, 255, Imgproc.THRESH_BINARY);
            mGrayTemplateImages.add(grayImage.clone());
        }

        //mDetector = FeatureDetector.create(FeatureDetector.ORB);
        //mExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        mMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        for(Mat tmpImage : mGrayTemplateImages) {
            MatOfKeyPoint trainKeypoints = new MatOfKeyPoint();
            Mat trainDescriptors = new Mat();
            akaze.detect(tmpImage, trainKeypoints);
            akaze.compute(tmpImage, trainKeypoints, trainDescriptors);
            /*
            mDetector.detect(tmpImage, trainKeypoints);
            mExtractor.compute(tmpImage, trainKeypoints, trainDescriptors);
            */
            mTrainDescriptorses.add(trainDescriptors);
        }

        mMatcher.add(mTrainDescriptorses);



    }

    @Override
    public void onResume() {
        super.onResume();

        running = true;
        mThread = new Thread(this);
        mThread.start();
    }

    @Override
    public void onPause() {
        running = false;
        while(true) {
            try {
                mThread.join();
                break;
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        super.onPause();
    }
}
