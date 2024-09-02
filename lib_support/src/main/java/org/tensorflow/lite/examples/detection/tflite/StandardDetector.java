package org.tensorflow.lite.examples.detection.tflite;

import android.app.Activity;

import java.io.IOException;

import org.tensorflow.lite.examples.detection.tflite.Detector.Device;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

/**
 * This TensorFlowLite detector works with the float MobileNet model.
 */
public class StandardDetector extends Detector {

    // TODO: check values
    private final float IMAGE_MEAN = 127.5f;
    private final float IMAGE_STD = 127.5f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    // TODO: check values
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 1.0f;

    /**
     * Initializes a {@code StandardDetector}.
     *
     * @param activity
     */
    public StandardDetector(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() { return "model.tflite"; }

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
