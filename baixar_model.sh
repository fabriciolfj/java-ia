bash# cria pasta local
mkdir -p models/all-MiniLM-L6-v2

# baixa os arquivos necessários
curl -L "https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/main/model.onnx" \
     -o models/all-MiniLM-L6-v2/model.onnx

curl -L "https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/main/tokenizer.json" \
     -o models/all-MiniLM-L6-v2/tokenizer.json