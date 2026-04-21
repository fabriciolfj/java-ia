package com.github.javaia.ianativajava;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.util.Downloader;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

public class Simultation {

    static void main() throws IOException {
        String model = "tjake/Llama-3.2-1B-Instruct-JQ4";
        String workDir = "./models";

        String prompt = "What is a Java Development Kit?";

        File modelPath = new Downloader(workDir, model).huggingFaceModel();
        AbstractModel m = ModelSupport.loadModel(modelPath, DType.F32, DType.I8);

        PromptContext ctx;
        if (m.promptSupport().isPresent()) {
            ctx = m.promptSupport()
                    .get()
                    .builder()
                    .addSystemMessage("You are a helpful chatbot who writes short responses")
                    .addUserMessage(prompt)
                    .build();
        } else {
            ctx = PromptContext.of(prompt);

            IO.println("Prompt " + ctx.getPrompt());
            Generator.Response r = m.generate(UUID.randomUUID(), ctx, 0.0f, 256, (s, f) -> {});

            IO.println(r.responseText);
        }
    }
}
