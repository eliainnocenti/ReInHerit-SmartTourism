package org.tensorflow.lite.examples.classification.tflite;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DatabaseAccess {
    private SQLiteOpenHelper openHelper;
    private SQLiteDatabase database;
    private static DatabaseAccess instance;

    /**
     * Private constructor to aboid object creation from outside classes.
     *
     * @param context
     */
    private DatabaseAccess(Context context) {
        this.openHelper = new DatabaseOpenHelper(context);
    }

    /**
     * Return a singleton instance of DatabaseAccess.
     *
     * @param context the Context
     * @return the instance of DabaseAccess
     */
    public static DatabaseAccess getInstance(Context context) {
        if (instance == null) {
            instance = new DatabaseAccess(context);
        }
        return instance;
    }

    /**
     * Open the database connection.
     */
    public void open() {
        this.database = openHelper.getWritableDatabase();
    }

    /**
     * Close the database connection.
     */
    public void close() {
        if (database != null) {
            this.database.close();
        }
    }

    /**
     * Read all quotes from the database.
     *
     * @return a List of quotes
     */
    public List<String> getElements() {
        List<String> list = new ArrayList<>();
        Cursor cursor = database.rawQuery("SELECT rowid,style,color FROM Elements", null);
        cursor.moveToFirst();
        while (!cursor.isAfterLast()) {
            list.add(cursor.getString(0));
            list.add(cursor.getString(1));
            list.add(cursor.getString(2));
            cursor.moveToNext();
        }
        cursor.close();
        return list;
    }

    public List<String> getMatrix(int i) {
        List<String> list = new ArrayList<>();
        Cursor cursor = database.rawQuery("SELECT value FROM Matrix WHERE element = " + i, null);
        cursor.moveToFirst();
        while (!cursor.isAfterLast()) {
            list.add(cursor.getString(0));
            cursor.moveToNext();
        }
        cursor.close();
        return list;
    }

    public ArrayList<Element> getFeatureDistance(float[] features) {
        ArrayList<Element> list = new ArrayList<>();
        Cursor cursor = database.rawQuery("SELECT * FROM AllInOne WHERE rowid < (SELECT COUNT(*) FROM AllInOne)/10", null);
        cursor.moveToFirst();
        while (!cursor.isAfterLast()) {
            String style = cursor.getString(0);
            String color =cursor.getString(1);
            String matrix = cursor.getString(2);

            //Convert matrix string to Float
            String[] splitted = matrix.substring(1,matrix.length() - 1).split("\\s+");
            ArrayList<Float> listMatrix = new ArrayList<Float>();

            for (String s: splitted
                 ) {
                listMatrix.add(Float.parseFloat(s));
            }

            //Calculate Distance
            double distance = euclideanDistance(features,listMatrix);

            Element e = new Element(style,color,distance);
            list.add(e);

            cursor.moveToNext();
        }
        cursor.close();
        return list;

    }

    private double euclideanDistance(float[] a, ArrayList<Float> b) {
        double diff_square_sum = 0.0;
        for (int i = 0; i < b.size(); i++) {
            diff_square_sum += (a[i] - b.get(i)) * (a[i] - b.get(i));
        }
        return Math.sqrt(diff_square_sum);
    }
}