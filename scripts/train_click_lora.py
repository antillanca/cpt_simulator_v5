import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer
from transformers import TrainingArguments

# 1. Configuración
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Base model for Qwen3:0.6b
DATASET_PATH = "dpo_dataset.jsonl"
OUTPUT_DIR = "click_adapter"
GGUF_NAME = "qwen3-iat"

def train():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} no encontrado. Ejecuta el modo Onda primero.")
        return

    # 2. Cargar Dataset
    data = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    dataset = Dataset.from_list(data)
    
    # 3. Cargar Modelo con Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # Agregar adaptadores LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    # 4. Configurar DPO Trainer
    PatchDPOTrainer()
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Ajustar según tamaño del dataset
            learning_rate = 5e-5,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = OUTPUT_DIR,
        ),
        train_dataset = dataset,
        tokenizer = tokenizer,
    )

    # 5. Entrenar
    print("Iniciando Fase de Colapso (Entrenamiento DPO)...")
    dpo_trainer.train()

    # 6. Guardar y Exportar a GGUF
    print(f"Exportando a {GGUF_NAME}.gguf...")
    model.save_pretrained_gguf(GGUF_NAME, tokenizer, quantization_method = "q4_k_m")
    print("¡Modo Click consolidado con éxito!")

if __name__ == "__main__":
    train()
