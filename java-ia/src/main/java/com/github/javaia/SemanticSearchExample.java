package com.github.javaia;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public class SemanticSearchExample {

    // base de textos (simula seu pgvector/Qdrant)
    static final List<String> BASE_TEXTOS = List.of(
            "financiamento de veículo zero km",
            "crédito para compra de automóvel usado",
            "empréstimo pessoal sem garantia",
            "refinanciamento de imóvel residencial",
            "consórcio para motocicleta"
    );

    public static void main(String[] args) throws Exception {

        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelPath(Paths.get("models/all-MiniLM-L6-v2"))
                .optModelName("model.onnx")
                .optEngine("OnnxRuntime")
                .optArgument("includeTokenTypes", "true")
                .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
             Predictor<String, float[]> predictor = model.newPredictor()) {

            // 1. gera embeddings para todos os textos da base
            System.out.println("=== Gerando embeddings da base ===");
            List<float[]> embeddings = predictor.batchPredict(BASE_TEXTOS);

            // 2. query do usuário
            String query = "quero financiar meu carro";
            System.out.println("\nQuery: " + query);

            // 3. gera embedding da query
            float[] queryEmbedding = predictor.predict(query);

            // 4. calcula cosine similarity com cada texto da base
            System.out.println("\n=== Similaridades ===");
            int maisSimlarIdx = 0;
            float maiorSimilaridade = -1;

            for (int i = 0; i < BASE_TEXTOS.size(); i++) {
                float similarity = cosineSimilarity(queryEmbedding, embeddings.get(i));
                System.out.printf("%.4f → %s%n", similarity, BASE_TEXTOS.get(i));

                if (similarity > maiorSimilaridade) {
                    maiorSimilaridade = similarity;
                    maisSimlarIdx = i;
                }
            }

            // 5. retorna o TEXTO mais similar
            System.out.println("\n=== Resultado ===");
            System.out.println("Texto mais similar: " + BASE_TEXTOS.get(maisSimlarIdx));
            System.out.printf("Score: %.4f%n", maiorSimilaridade);
        }
    }

    static float cosineSimilarity(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }
}

/**
 *O model.onnx só sabe fazer contas com tensores — ele não sabe o que é texto. O tokenizer.json é quem faz a ponte entre texto e números.
 *
 * criteria.loadModel()
 *         ↓
 * TextEmbeddingTranslatorFactory.newInstance()
 *         ↓
 * lê tokenizer.json da mesma pasta do model.onnx
 *         ↓
 * cria HuggingFaceTokenizer internamente
 *         ↓
 * cria TextEmbeddingTranslator (pre + post processing)
 *
 *
 * predictor.predict("financiamento de veículo")
 *         ↓
 * [PRE-PROCESSING]  ← TextEmbeddingTranslator.processInput()
 *   HuggingFaceTokenizer lê tokenizer.json
 *   "financiamento de veículo" → [101, 8275, 1997, 2744, 102]
 *   monta os tensors: input_ids, attention_mask, token_type_ids
 *         ↓
 * [INFERÊNCIA]  ← model.onnx executa
 *   tensors entram no modelo ONNX
 *   modelo faz as contas (384 dimensões)
 *         ↓
 * [POST-PROCESSING]  ← TextEmbeddingTranslator.processOutput()
 *   pega tensor de saída
 *   aplica mean pooling
 *   aplica normalização L2
 *   converte para float[]
 *         ↓
 * [0.6, -0.45, 0.8, ...]
 *
 *models/all-MiniLM-L6-v2/
 * ├── model.onnx       ← carregado pelo OnnxRuntime (inferência)
 * └── tokenizer.json   ← carregado pelo TextEmbeddingTranslatorFactory (pre/post processing)
 *
 *
 */