#!/bin/bash
# Despliega el modelo consolidado en Ollama

GGUF_FILE="qwen3-iat/Qwen2.5-0.5B-Q4_K_M.gguf"

if [ ! -d "qwen3-iat" ]; then
    # Unsloth a veces guarda el archivo con otro nombre o estructura
    # Buscamos el .gguf generado
    GGUF_FILE=$(find . -name "*.gguf" | head -n 1)
fi

if [ -z "$GGUF_FILE" ]; then
    echo "Error: No se encontró el archivo GGUF. Ejecuta primero train_click_lora.py"
    exit 1
fi

echo "Creando Modelfile para Ollama..."
cat << MODEOF > Modelfile
FROM $GGUF_FILE
SYSTEM """You are the i@ Observer Agent. You are an expert in physics and mathematics logic for the CPT Simulator."""
MODEOF

echo "Registrando modelo qwen3-iat en Ollama..."
ollama create qwen3-iat -f Modelfile

echo "¡Modelo qwen3-iat desplegado!"
ollama list | grep qwen3-iat
