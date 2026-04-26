package com.github.javaia.claudeia;

import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Model;

public class UseApiSimulation {

    static void main() {
        AnthropicClient client = AnthropicOkHttpClient.fromEnv();

        MessageCreateParams params = MessageCreateParams.builder()
                .maxTokens(1024L)
                .addUserMessage("Olá, Claude!")
                .model(Model.CLAUDE_SONNET_4_6)
                .build();

        Message message = client.messages().create(params);
        final String texto = message.content().stream()
                        .filter(block -> block.isText())
                                .map(contentBlock -> contentBlock.asText().text())
                                        .findFirst()
                                                .orElse("");

        IO.println(texto);
    }
}
