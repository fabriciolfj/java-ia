package com.github.javaia;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.net.http.*;
import java.util.List;
import java.util.Map;

public class OllamaEmbeddingExample {

    private static final String OLLAMA_URL = "http://localhost:11434/api/embed";
    private static final String MODEL = "nomic-embed-text";

    public static void main(String[] args) throws Exception {

        String texto = "O Pix é um sistema de pagamento instantâneo brasileiro";

        float[] embedding = gerarEmbedding(texto);

        System.out.println("Dimensões: " + embedding.length); // 768
        System.out.printf("Primeiros valores: [%.4f, %.4f, %.4f, ...]%n",
                embedding[0], embedding[1], embedding[2]);
    }

    public static float[] gerarEmbedding(String texto) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        HttpClient client = HttpClient.newHttpClient();
        String body = mapper.writeValueAsString(Map.of(
                "model", MODEL,
                "input", texto
        ));


        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(OLLAMA_URL))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response = client.send(request,
                HttpResponse.BodyHandlers.ofString());

        // resposta: { "embeddings": [[0.12, 0.87, ...]] }
        var root = mapper.readTree(response.body());
        var array = root.get("embeddings").get(0);

        float[] vector = new float[array.size()];
        for (int i = 0; i < array.size(); i++) {
            vector[i] = array.get(i).floatValue();
        }
        return vector;
    }
}