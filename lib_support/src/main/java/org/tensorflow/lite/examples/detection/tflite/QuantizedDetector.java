package org.tensorflow.lite.examples.detection.tflite;

import android.app.Activity;

import java.io.IOException;

import org.tensorflow.lite.examples.detection.tflite.Detector.Device;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

/**
 * This TensorFlow Lite detector works with the quantized MobileNet model.
 */
public class QuantizedDetector extends Detector {

    /**
     * The quantized model does not require normalization, thus set mean as 0.0f, and std as 1.0f to
     * bypass the normalization.
     */
    // TODO: check values
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;

    /**
     * Quantized MobileNet requires additional dequantization to the output probability.
     */
    // TODO: check values
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;

    /**
     * Initializes a {@code QuantizedDetector}.
     *
     * @param activity
     */
    public QuantizedDetector(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() { return "model_fp16.tflite"; }

    @Override
    protected String getLabelPath() { return "model.txt"; }

    @Override
    protected String getLabel(int classIndex) {
        return DetectorConstants.LABELS[classIndex];
    }

    @Override
    protected float getDetectionThreshold(int classIndex) {
        return DetectorConstants.DETECTION_THRESHOLDS[classIndex];
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
}
