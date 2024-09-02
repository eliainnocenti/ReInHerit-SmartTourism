package org.tensorflow.lite.examples.detection.tflite;

public class DetectorConstants {

    public static final String[] LABELS = {
            "background",
            "Cattedrale di Santa Maria del Fiore",
            "Battistero di San Giovanni",
            "Campanile di Giotto",
            "Galleria degli Uffizi",
            "Loggia dei Lanzi",
            "Palazzo Vecchio",
            "Ponte Vecchio",
            "Basilica di Santa Croce",
            "Palazzo Pitti",
            "Piazzale Michelangelo",
            "Basilica di Santa Maria Novella",
            "Basilica di San Miniato al Monte"
    };

    // TODO: update values
    public static final float[] DETECTION_THRESHOLDS = {
            0.5f, // background // TODO: check
            0.7f, // santamariadelfiore
            0.6f, // battisterosangiovanni
            0.8f, // campanilegiotto
            0.6f, // galleriauffizi
            0.5f, // loggialanzi
            0.7f, // palazzovecchio
            0.6f, // pontevecchio
            0.6f, // basilicasantacroce
            0.7f, // palazzopitti
            0.6f, // piazzalemichelangelo
            0.6f, // basilicasantamarianovella
            0.7f  // basilicasanminiato
    };

}
