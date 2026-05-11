package com.github.javaia;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import java.util.Arrays;

public class EmbeddingTest {

    static void main() {
        EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

        var responseCar = embeddingModel.embed("car");
        var responseCat = embeddingModel.embed("cat");
        var responseKitten = embeddingModel.embed("kitten");

        float[] carVector = responseCar.content().vector();
        float[] catVector = responseCat.content().vector();
        float[] kittenVector = responseKitten.content().vector();

        var vArrayCar = Nd4j.create(carVector);
        var vArrayCat = Nd4j.create(catVector);
        var vArrayKit = Nd4j.create(kittenVector);


        IO.println(CosineSimilarity.calculeConsineSimilariy(vArrayCar, vArrayCat));
        IO.println(CosineSimilarity.calculeConsineSimilariy(vArrayCar, vArrayKit));
        IO.println(CosineSimilarity.calculeConsineSimilariy(vArrayCat, vArrayKit));
    }
}
