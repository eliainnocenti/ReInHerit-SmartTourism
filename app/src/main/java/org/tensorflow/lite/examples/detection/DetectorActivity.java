package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.Image;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

//import org.tensorflow.lite.examples.detection.R;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.Detector.Device;
import org.tensorflow.lite.examples.detection.tflite.Detector.Model;

public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    //private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final Size DESIRED_PREVIEW_SIZE = new Size(384, 384);

    private static final float TEXT_SIZE_DIP = 10;

    private Bitmap rgbFrameBitmap = null;
    private long lastProcessingTimeMs;
    private Integer sensorOrientation;
    private Detector detector;
    private BorderedText borderedText;

    /**
     * Input image size of the model along x axis.
     */
    private int imageSizeX;
    /**
     * Input image size of the model along y axis.
     */
    private int imageSizeY;

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_ic_camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        recreateDetector(getModel(), getDevice(), getNumThreads());
        if (detector == null) {
            LOGGER.e("No detector on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight); // 640 X 480

        //rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        //rgbFrameBitmap = Bitmap.createBitmap(imageSizeX, imageSizeY, Config.ARGB_8888);
        rgbFrameBitmap = Bitmap.createBitmap(384, 384, Config.ARGB_8888);
    }

    // DEBUG
    private void saveBitmapToFile(Bitmap bitmap, String filename) {
        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        File file = new File(path, filename);

        try {
            FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            out.flush();
            out.close();
            LOGGER.d("saving", "Saved frame to " + file.getAbsolutePath());
        } catch (Exception e) {
            LOGGER.e("saving", "Error saving frame", e);
        }
    }

    // DEBUG
    private int frameCount = 0;
    private static final int SAVE_FRAME_INTERVAL = 10; // Salva un frame ogni 30 frame
    private static final int MAX_FRAMES_TO_SAVE = 50;
    private int savedFramesCount = 0;

    @Override
    protected void processImage() {

        LOGGER.d("processImage: Start image processing");

        // Ensure rgbFrameBitmap is initialized
        if (rgbFrameBitmap == null) {
            LOGGER.d("processImage: rgbFrameBitmap initialization");
            rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        }

        int[] rgbBytes = getRgbBytes();
        LOGGER.d("processImage: rgbBytes array length : " + rgbBytes.length);

        try {
            // Make sure these values don't exceed the bitmap's dimensions
            rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
            LOGGER.d("processImage: Pixel impostati su rgbFrameBitmap");
        } catch (IllegalArgumentException e) {
            String logTag = "BitmapProcessor"; // Or use a relevant class name
            LOGGER.e(logTag, "Error setting bitmap pixels. Bitmap dimensions: " +
                    previewWidth + "x" + previewHeight +
                    ", RGB byte array length:" + rgbBytes.length, e);
            // Handle the error, maybe by resizing the input or adjusting the preview size
        }

        final int cropSize = Math.min(previewWidth, previewHeight);
        LOGGER.d("processImage: Crop dimension: " + cropSize);

        // DEBUG
        //Bitmap bitmap_test1 = BitmapFactory.decodeResource(getResources(), R.drawable.basilicasantacroce);
        //Bitmap bitmap_test2 = BitmapFactory.decodeResource(getResources(), R.drawable.battisterosangiovanni); // OK
        //Bitmap bitmap_test3 = BitmapFactory.decodeResource(getResources(), R.drawable.campanilegiotto);
        //Bitmap bitmap_test4 = BitmapFactory.decodeResource(getResources(), R.drawable.palazzovecchio);
        //Bitmap bitmap_test5 = BitmapFactory.decodeResource(getResources(), R.drawable.santamariadelfiore);    // OK

        // FIXME
        // Cattedrale di Santa Maria del Fiore
        //Bitmap bitmap_test6 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_santamariadelfiore_0093); // Front // OK
        //Bitmap bitmap_test7 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_santamariadelfiore_0095); // Side  // NOT OK
        //Bitmap bitmap_test8 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_santamariadelfiore_0096); // Back  // OK

        // FIXME
        // Battistero di San Giovanni
        //Bitmap bitmap_test9 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_battisterosangiovanni_0090);  // NOT OK
        //Bitmap bitmap_test10 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_battisterosangiovanni_0097); // OK 0.7

        // FIXME
        // Campanile di Giotto
        //Bitmap bitmap_test11 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_campanilegiotto_0093); // NOT OK
        //Bitmap bitmap_test12 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_campanilegiotto_0095); // NOT OK
        //Bitmap bitmap_test13 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_campanilegiotto_0098); // NOT OK

        // FIXME
        // Palazzo Vecchio
        //Bitmap bitmap_test14 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_palazzovecchio_0090); // OK
        //Bitmap bitmap_test15 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_palazzovecchio_0094); // NOT OK
        //Bitmap bitmap_test16 = BitmapFactory.decodeResource(getResources(), R.drawable.florence_palazzovecchio_0095); // NOT OK

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        if (detector != null) {

                            /* // DEBUG
                            frameCount++;
                            LOGGER.e("processImage: Frame count: " + frameCount + " - SAVE_FRAME_INTERVAL: " + SAVE_FRAME_INTERVAL + " - %: " + (frameCount % SAVE_FRAME_INTERVAL));
                            if (frameCount % SAVE_FRAME_INTERVAL == 0 && savedFramesCount < MAX_FRAMES_TO_SAVE) {
                                saveBitmapToFile(rgbFrameBitmap, "frame_" + frameCount + ".png");
                                savedFramesCount++;
                                LOGGER.e("processImage: Saved frame" + frameCount + ".png");
                            }
                            */

                            // Resize rgbFrameBitmap to match the model input size
                            LOGGER.d("processImage: Start bitmap resizingp");
                            //Bitmap resizedBitmap = Bitmap.createScaledBitmap(rgbFrameBitmap, imageSizeX, imageSizeY, true);
                            //Bitmap resizedBitmap = Bitmap.createScaledBitmap(rgbFrameBitmap, 384, 384, true);

                            // DEBUG
                            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap_test16, 384, 384, true);

                            LOGGER.d("processImage: Bitmap resized to 384x384");

                            LOGGER.d("Camera Image Dimensions: " + previewWidth + "x" + previewHeight);
                            LOGGER.d("Resized Image Dimensions: " + resizedBitmap.getWidth() + "x" + resizedBitmap.getHeight());
                            LOGGER.d("Sensor Orientation: " + sensorOrientation);

                            final long startTime = SystemClock.uptimeMillis();
                            LOGGER.d("processImage: Start image recognition");
                            final List<Detector.Detection> results = detector.recognizeImage(resizedBitmap, sensorOrientation);
                            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                            LOGGER.d("processImage: Processing time: " + lastProcessingTimeMs + "ms");
                            LOGGER.d("processImage: Number of detections: " + results.size());

                            runOnUiThread(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            showResultsInBottomSheet(results);
                                            showFrameInfo(previewWidth + "x" + previewHeight);
                                            showCropInfo(imageSizeX + "x" + imageSizeY);
                                            showCameraResolution(cropSize + "x" + cropSize);
                                            showRotationInfo(String.valueOf(sensorOrientation));
                                            showInference(lastProcessingTimeMs + "ms");
                                            LOGGER.d("processImage: Results shown in UI");
                                        }
                                    });

                        } else {
                            LOGGER.w("processImage: Detector is null");
                        }
                        readyForNextImage();
                        LOGGER.d("processImage: Ready for next image");
                    }
                });
    }

    @Override
    protected void onInferenceConfigurationChanged() {
        if (rgbFrameBitmap == null) {
            // Defer creation until we're getting camera frames.
            return;
        }
        final Device device = getDevice();
        final Model model = getModel();
        final int numThreads = getNumThreads();
        runInBackground(() -> recreateDetector(model, device, numThreads));
    }

    private void recreateDetector(Model model, Device device, int numThreads) {
        if (detector != null) {
            LOGGER.d("Closing detector.");
            detector.close();
            detector = null;
        }
        if (device == Detector.Device.GPU
                && (model == Detector.Model.QUANTIZED)) {
            LOGGER.d("Not creating detector: GPU doesn't support quantized models.");
            runOnUiThread(
                    () -> {
                        Toast.makeText(this, R.string.tfe_ic_gpu_quant_error, Toast.LENGTH_LONG).show();
                    });
            return;
        }
        try {
            LOGGER.d(
                    "Creating detector (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
            detector = Detector.create(this, model, device, numThreads);
        } catch (IOException | IllegalArgumentException e) {
            LOGGER.e(e, "Failed to create detector.");
            runOnUiThread(
                    () -> {
                        Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                    });
            return;
        }

        // Updates the input image size.
        imageSizeX = detector.getImageSizeX();
        imageSizeY = detector.getImageSizeY();
    }
}
