package org.tensorflow.lite.examples.detection;

import static com.google.common.truth.Truth.assertThat;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.rule.ActivityTestRule;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.examples.classification.ClassifierActivity;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.Detector.Device;
import org.tensorflow.lite.examples.detection.tflite.Detector.Model;
import org.tensorflow.lite.examples.detection.tflite.Detector.Detection;
import org.tensorflow.lite.examples.detection.tflite.StandardDetector;

/** Golden test for Image Classification Reference app. */
@RunWith(AndroidJUnit4.class)
public class DetectorTest {

    private Detector detector;

    @Rule
    public ActivityTestRule<DetectorActivity> rule =
            new ActivityTestRule<>(DetectorActivity.class);

    @Before
    public void setUp() throws Exception {

        DetectorActivity activity = rule.getActivity();
        // Inizializza il Detector
        detector = new StandardDetector(
                activity,
                Detector.Device.CPU,
                1 // Numero di thread
        );
    }

    @Test
    public void testRecognizeImage() {
        // Carica un'immagine di test dalla cartella drawable
        Bitmap testBitmap = BitmapFactory.decodeResource(
                InstrumentationRegistry.getInstrumentation().getContext().getResources(),
                R.drawable.battisterosangiovanni
        );

        // Assumi che l'orientamento della fotocamera sia 270 gradi
        int sensorOrientation = 270; // TODO: check

        // Chiama il metodo di riconoscimento
        List<Detector.Detection> detections = detector.recognizeImage(testBitmap, sensorOrientation);

        // Verifica che il risultato non sia nullo
        assertNotNull("Detections should not be null", detections);

        // Verifica che ci siano risultati
        assertTrue("There should be at least one detection", detections.size() > 0);

        // Aggiungi ulteriori asserzioni basate sui tuoi criteri di test
        for (Detector.Detection detection : detections) {
            Log.d("DetectorTest", "Detection: " + detection);
            assertTrue("Confidence should be greater than 0", detection.getConfidence() > 0);
        }
    }

}
