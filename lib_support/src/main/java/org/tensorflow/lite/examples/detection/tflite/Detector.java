package org.tensorflow.lite.examples.detection.tflite;

import static java.lang.Math.min;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A wrapper class for a TensorFlow Lite object detection model.
 */
public abstract class Detector {
    private static final String TAG = "ObjectDetector";

    /**
     * Image size along the x axis.
     */
    private final int imageSizeX;
    /**
     * Image size along the y axis.
     */
    private final int imageSizeY;

    /** The loaded TensorFlow Lite model. */
    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    private Interpreter tflite;

    /** Labels corresponding to the output of the vision model. */
    private final List<String> labels;
    /**
     * Optional GPU delegate for accleration.
     */
    private GpuDelegate gpuDelegate = null;
    /**
     * Optional NNAPI delegate for accleration.
     */
    private NnApiDelegate nnApiDelegate = null;
    /**
     * Input image TensorBuffer.
     */
    private TensorImage inputImageBuffer;

    private int numOutputs;

    // Output results
    private float[][][] outputScores;
    private float[][][] outputBoxes;

    private final Context context;

    /**
     * Initializes a {@code Classifier}.
     */
    protected Detector(Activity activity, Device device, int numThreads) throws IOException {

        this.context = activity.getApplicationContext();

        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
        switch (device) {
            case NNAPI:
                nnApiDelegate = new NnApiDelegate();
                tfliteOptions.addDelegate(nnApiDelegate);
                break;
            case GPU:
                CompatibilityList compatList = new CompatibilityList();
                if (compatList.isDelegateSupportedOnThisDevice()) {
                    // if the device has a supported GPU, add the GPU delegate
                    GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                    GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                    tfliteOptions.addDelegate(gpuDelegate);
                    Log.d(TAG, "GPU supported. GPU delegate created and added to options");
                } else {
                    tfliteOptions.setUseXNNPACK(true);
                    Log.d(TAG, "GPU not supported. Default to CPU.");
                    Toast.makeText(context, "GPU not supported. Default to CPU.", Toast.LENGTH_SHORT);
                }
                break;
            case CPU:
                tfliteOptions.setUseXNNPACK(true);
                Log.d(TAG, "CPU execution");
                break;
        }
        tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        // Loads labels out from the label file.
        // TODO: check
        labels = FileUtil.loadLabels(activity, getLabelPath());
        //labels = Arrays.asList(FileUtil.loadLabels(activity, getLabelPath()));

        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];

        // Determine the number of outputs dynamically
        numOutputs = tflite.getOutputTensorCount();
        Log.d(TAG, "Number of output tensors: " + numOutputs);

        // Initialize output arrays
        int[] boxesShape = tflite.getOutputTensor(0).shape();
        outputBoxes = new float[boxesShape[0]][boxesShape[1]][boxesShape[2]];

        int[] scoresShape = tflite.getOutputTensor(1).shape();
        outputScores = new float[scoresShape[0]][scoresShape[1]][scoresShape[2]];

        inputImageBuffer = new TensorImage(tflite.getInputTensor(0).dataType());

        Log.d(TAG, "Created a Tensorflow Lite Object Detector.");
    }

    /**
     * Creates a detector with the provided configuration.
     *
     * @param activity   The current Activity.
     * @param model      The model to use for detection.
     * @param device     The device to use for detection.
     * @param numThreads The number of threads to use for detection.
     * @return A detector with the desired configuration.
     */
    public static Detector create(Activity activity, Model model, Device device, int numThreads)
            throws IOException {
        if (model == Model.STANDARD) {
            return new StandardDetector(activity, device, numThreads);
        } else if (model == Model.QUANTIZED) {
            return new QuantizedDetector(activity, device, numThreads);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    // TODO: do I need any other methods?
    // createMap

    // TODO: can I encapsulate some methods?

    /**
     * Runs inference and returns the detection results.
     */
    public List<Detection> recognizeImage(Bitmap bitmap, int sensorOrientation) {
        // Logs this method so that it can be analyzed with systrace.

        Trace.beginSection("recognizeImage");

        Log.d(TAG, "recognizeImage: Start image recognition");

        //Load image
        Trace.beginSection("loadImage");
        Log.d(TAG, "recognizeImage: Start image loading");
        long startTimeForLoadImage = SystemClock.uptimeMillis();

        // TODO: check loadImage
        inputImageBuffer = loadImage(bitmap, sensorOrientation, 1f); // Load image (1 + 0 for augumentation)

        Log.d("DetectorActivity", "loadImage: Image loaded");

        long endTimeForLoadImage = SystemClock.uptimeMillis();
        Log.d(TAG, "recognizeImage: Image loading time: " + (endTimeForLoadImage - startTimeForLoadImage) + "ms");

        Trace.endSection();
        //Log.v(TAG, "Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

        // Runs the inference call.
        Trace.beginSection("runInference");
        Log.d(TAG, "recognizeImage: Start inference");
        long startTime = SystemClock.uptimeMillis();
        Object[] inputArray = {inputImageBuffer.getBuffer()};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputBoxes);  // TODO: check 0/1
        outputMap.put(1, outputScores); // TODO: check 0/1
        tflite.runForMultipleInputsOutputs(inputArray, outputMap);
        long endTime = SystemClock.uptimeMillis();
        Trace.endSection();
        Log.d(TAG, "recognizeImage: Inference time: " + (endTime - startTime) + "ms");
        //Log.v(TAG, "Timecost to run model inference: " + (endTime - startTime) + " ms");

        // Collect and return results
        List<Detection> detections = new ArrayList<>();

        // TODO: is xmin, ymin, xmax, ymax or xmin, ymin, width, height?

        int numDetections = outputBoxes[0].length;
        int numClasses = outputScores[0][0].length;
        Log.d(TAG, "recognizeImage: Number of detections: " + numDetections + ", Number of classes: " + numClasses);

        // Iterate over all the detections (boxes) generated by the model.
        for (int i = 0; i < numDetections; i++) {
            // Iterate over all classes (except background) for the current box.
            for (int classIndex = 1; classIndex < numClasses; classIndex++) {
                float score = outputScores[0][i][classIndex];

                // DEBUG
                if (score > 0.5f) {
                    //Log.v(TAG, "Detection - index: " + i + ", classIndex: " + classIndex + ", score: " + score);
                    Log.v(TAG, "recognizeImage: Detection - index: " + i + ", classIndex: " + classIndex + ", score: " + score);
                }

                // Filter results based on the threshold for that class.
                if (score >= getDetectionThreshold(classIndex)) {
                    // Box coordinates: xmin, ymin, xmax, ymax (normalized).
                    float xminNormalized = outputBoxes[0][i][0];
                    float yminNormalized = outputBoxes[0][i][1];
                    float xmaxNormalized = outputBoxes[0][i][2]; // TODO: check if 'xmax' or 'width'
                    float ymaxNormalized = outputBoxes[0][i][3]; // TODO: check if 'ymax' or 'height'

                    Log.v(TAG, "recognizeImage: DetectionTreshold - class: " + getLabel(classIndex) +
                            " - treshold: " + getDetectionThreshold(classIndex) + " - score: " + score);

                    int bitmapWidth = bitmap.getWidth();
                    int bitmapHeight = bitmap.getHeight();

                    // Denormalize coordinates using the dimensions of the bitmap.
                    float xmin = xminNormalized * bitmapWidth;
                    float ymin = yminNormalized * bitmapHeight;
                    float xmax = xmaxNormalized * bitmapWidth;
                    float ymax = ymaxNormalized * bitmapHeight;

                    // Create a RectF object for the position.
                    RectF location = new RectF(xmin, ymin, xmax, ymax);

                    // Create a new Detection object and add it to the list.
                    detections.add(new Detection(
                            String.valueOf(classIndex),  // Class ID
                            getLabel(classIndex),        // Class Title
                            score,                       // Confidence
                            location                     // Location
                    ));

                    Log.d(TAG, "recognizeImage: Added detection - Class: " + getLabel(classIndex) + ", Score: " + score);

                    // DEBUG
                    //Log.v(TAG, "Detection Box - xmin: " + xmin + ", ymin: " + ymin + ", xmax: " + xmax + ", ymax: " + ymax);
                }
            }
        }

        Log.d(TAG, "recognizeImage: Total detections after filtering: " + detections.size());

        Trace.endSection(); // end recognize image section

        return detections;

        /*
        for (int i = 0; i < numDetections; i++) {
            if (outputScores[0][i] > getDetectionThreshold((int) outputClasses[0][i])) {
                float ymin = Math.max(0, outputBoxes[0][i][0][0]) * bitmap.getHeight();
                float xmin = Math.max(0, outputBoxes[0][i][0][1]) * bitmap.getWidth();
                float ymax = Math.min(1, outputBoxes[0][i][0][2]) * bitmap.getHeight();
                float xmax = Math.min(1, outputBoxes[0][i][0][3]) * bitmap.getWidth();
                RectF location = new RectF(xmin, ymin, xmax, ymax);
                detections.add(new Detection(
                        "" + i,
                        getLabel((int) outputClasses[0][i]),
                        outputScores[0][i],
                        location));
            }
        }
        */
        /*
        for (int i = 0; i < numDetections; i++) {
            if (outputScores[0][i] > getDetectionThreshold(i)) { // FIXME
                float ymin = Math.max(0, outputBoxes[0][i][0]) * bitmap.getHeight();
                float xmin = Math.max(0, outputBoxes[0][i][1]) * bitmap.getWidth();
                float ymax = Math.min(1, outputBoxes[0][i][2]) * bitmap.getHeight();
                float xmax = Math.min(1, outputBoxes[0][i][3]) * bitmap.getWidth();
                RectF location = new RectF(xmin, ymin, xmax, ymax);
                detections.add(new Detection(
                        "" + i,
                        getLabel(i),
                        outputScores[0][i], // FIXME
                        location));
            }
        }
        */
        /*
        for (int i = 0; i < numDetections; i++) {
            if (outputScores[0][i][0] > getDetectionThreshold(0)) { // Assumiamo che la classe 0 sia il background
                float ymin = Math.max(0, outputBoxes[0][i][0]) * bitmap.getHeight();
                float xmin = Math.max(0, outputBoxes[0][i][1]) * bitmap.getWidth();
                float ymax = Math.min(1, outputBoxes[0][i][2]) * bitmap.getHeight();
                float xmax = Math.min(1, outputBoxes[0][i][3]) * bitmap.getWidth();
                RectF location = new RectF(xmin, ymin, xmax, ymax);

                // Trova la classe con il punteggio più alto
                int bestClassIndex = 0;
                float bestScore = outputScores[0][i][0];
                for (int j = 1; j < 13; j++) {
                    if (outputScores[0][i][j] > bestScore) {
                        bestScore = outputScores[0][i][j];
                        bestClassIndex = j;
                    }
                }

                Log.v(TAG, "Detection:" + i + " " + getLabel(bestClassIndex) + " " + bestScore);

                detections.add(new Detection(
                        "" + i,
                        getLabel(bestClassIndex),
                        bestScore,
                        location));
            }
        }
        */
        /*
        Log.e(TAG, "numDetections: " + numDetections);

        for (int i = 0; i < numDetections; i++) {
            // Trova la classe con il punteggio più alto
            int bestClassIndex = 0;
            float bestScore = outputScores[0][i][0];
            for (int j = 1; j < outputScores[0][i].length; j++) {
                if (outputScores[0][i][j] > bestScore) {
                    bestScore = outputScores[0][i][j];
                    bestClassIndex = j;
                }
            }

            // Applica una soglia di confidenza (ad esempio 0.5)
            if (bestScore > 0.1f) {

                // print best score with label
                Log.v(TAG, "Detection:" + i + " " + getLabel(bestClassIndex) + " " + bestScore);

                float[] boxCoords = outputBoxes[0][i];
                RectF location = new RectF(
                        boxCoords[1] * bitmap.getWidth(),
                        boxCoords[0] * bitmap.getHeight(),
                        boxCoords[3] * bitmap.getWidth(),
                        boxCoords[2] * bitmap.getHeight()
                );

                Detection detection = new Detection(
                        "" + i,
                        getLabel(bestClassIndex),
                        bestScore,
                        location
                );
                detections.add(detection);

                Log.v(TAG, "Detection: " + i + " " + getLabel(bestClassIndex) + " " + bestScore);
            }

        }
        */
    }

    /*
    public List<Detection> detect(Bitmap bitmap, int sensorOrientation) {
        inputImageBuffer = loadImage(bitmap, sensorOrientation, 1f);

        tflite.run(inputImageBuffer.getBuffer(), outputBoxesBuffer.getBuffer().rewind());
        tflite.run(inputImageBuffer.getBuffer(), outputScoresBuffer.getBuffer().rewind());

        return postprocessDetections(outputBoxesBuffer, outputScoresBuffer);
    }

    private List<Detection> postprocessDetections(TensorBuffer boxesBuffer, TensorBuffer scoresBuffer) {
        float[] boxesArray = boxesBuffer.getFloatArray();
        float[] scoresArray = scoresBuffer.getFloatArray();
        List<Detection> detections = new ArrayList<>();

        float confidenceThreshold = 0.5f;

        for (int i = 0; i < scoresArray.length; i++) {
            if (scoresArray[i] > confidenceThreshold) {
                float x = boxesArray[i * 4];
                float y = boxesArray[i * 4 + 1];
                float width = boxesArray[i * 4 + 2];
                float height = boxesArray[i * 4 + 3];
                detections.add(new Detection(x, y, width, height, scoresArray[i]));
            }
        }

        return detections;
    }
    */

    /**
     * Closes the interpreter and model to release resources.
     */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
    }

    /**
     * Get the image size along the x axis.
     */
    public int getImageSizeX() {
        return imageSizeX;
    }

    /**
     * Get the image size along the y axis.
     */
    public int getImageSizeY() {
        return imageSizeY;
    }

    /**
     * Loads input image, and applies preprocessing.
     */
    private TensorImage loadImage(Bitmap bitmap, int sensorOrientation, float zoomRatio) {
        // Loads bitmap into a TensorImage.

        Log.d(TAG, "loadImage: Starting image loading");
        Log.d(TAG, "loadImage: Bitmap dimensions: " + bitmap.getWidth() + "x" + bitmap.getHeight());
        Log.d(TAG, "loadImage: Sensor orientation: " + sensorOrientation + ", Zoom ratio: " + zoomRatio);

        //inputImageBuffer.load(bitmap); //image in ARGB_8888

        //inputImageBuffer = new TensorImage(DataType.FLOAT32);
        inputImageBuffer.load(bitmap);

        Log.d(TAG, "loadImage: Bitmap loaded into inputImageBuffer");

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;

        int cropSizeZoom = (int) (cropSize * zoomRatio);

        Log.d(TAG, "loadImage: Crop size: " + cropSize + ", Num rotation: " + numRotation + ", Crop size zoom: " + cropSizeZoom);

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        //.add(new ResizeWithCropOrPadOp(cropSizeZoom, cropSizeZoom))

                        // To get the same inference results as lib_task_api, which is built on top of the Task
                        // Library, use ResizeMethod.BILINEAR. ERA (ResizeMethod.NEAREST_NEIGHBOR)
                        //.add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.BILINEAR)) // TODO: check

                        .add(new Rot90Op(numRotation))

                        .add(getPreprocessNormalizeOp())
                        //.add(new NormalizeOp(0.0f, 1.0f)) // Normalizes pixel values between 0 and 1 as in Python

                        .build();

        Log.d(TAG, "loadImage: ImageProcessor built");

        return imageProcessor.process(inputImageBuffer);
    }

    /**
     * Gets the name of the model file stored in Assets.
     */
    protected abstract String getModelPath();

    /**
     * Gets the name of the label file stored in Assets.
     */
    protected abstract String getLabelPath();

    /**
     * Gets the label for the given class index.
     */
    protected abstract String getLabel(int classIndex);

    /**
     * Gets the detection threshold for the given class index.
     */
    protected abstract float getDetectionThreshold(int classIndex);

    /**
     * Gets the TensorOperator to nomalize the input image in preprocessing.
     */
    protected abstract TensorOperator getPreprocessNormalizeOp();

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract TensorOperator getPostprocessNormalizeOp();

    /**
     * The model type used for detection.
     */
    public enum Model {
        STANDARD,
        QUANTIZED
    }

    /**
     * The runtime device type used for executing classification.
     */
    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    /**
     *
     */
    public enum Language {
        English,
        Italian
    }

    /**
     * An immutable result returned by a Detector describing what was detect.
     */
    public static class Detection {
        private final String id;
        private final String title;
        private final Float confidence;
        private RectF location;

        // TODO: do I need to set the parameters as "final" parameters?
        public Detection(String id, String title, Float confidence, RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        // Getters
        public String getId() { return id; }
        public String getTitle() { return title; }
        public Float getConfidence() { return confidence; }
        public RectF getLocation() { return new RectF(location); }

        @NonNull
        @Override
        public String toString() {
            return "Detection{" +
                    "id='" + id + '\'' +
                    ", title='" + title + '\'' +
                    ", confidence=" + confidence +
                    ", location=" + location +
                    '}';
        }
    }
}
