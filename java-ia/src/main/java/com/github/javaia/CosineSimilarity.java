package com.github.javaia;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class CosineSimilarity {

    public static double calculeConsineSimilariy(INDArray vectorA,
                                                 INDArray vectorB) {
        if (vectorA.length() != vectorB.length()) {
            throw new IllegalArgumentException("Vectors must have the same dimensions");
        }

        double dotProduct = Nd4j.getBlasWrapper().dot(vectorA, vectorB);

        double magnitudeA = vectorA.norm2Number().doubleValue();
        double magnitudeB = vectorB.norm2Number().doubleValue();

        if (magnitudeA == 0 || magnitudeB == 0) {
            return 0;
        }

        return dotProduct / (magnitudeA * magnitudeB);
    }

    static void main() {
        INDArray dog = Nd4j.create(new float[]{0.9f, 0.1f, 0.2f});
        INDArray cat = Nd4j.create(new float[]{0.8f, 0.2f, 0.1f});
        INDArray car = Nd4j.create(new float[]{0.1f, 0.9f, 0.8f});

        IO.println(calculeConsineSimilariy(dog, cat));
        IO.println(calculeConsineSimilariy(dog, car));
        IO.println(calculeConsineSimilariy(car, cat));
    }
}
