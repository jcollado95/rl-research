#!/usr/bin/env python3
"""
REINFORCE + LoRA para ajuste fino de un LLM en tareas de opción múltiple (MCQA)
================================================================================

Implementa el algoritmo REINFORCE (Williams, 1992), el método de policy
gradient más simple, para entrenar SmolLM2-135M-Instruct a responder
preguntas del dataset CommonsenseQA.

¿Por qué REINFORCE?
- Es el algoritmo de policy gradient más simple que existe.
- No necesita un modelo crítico (critic-free).
- No necesita un modelo de recompensa separado: usamos una recompensa
  basada en reglas (¿acertó la respuesta? → 1, ¿falló? → 0).

¿Por qué LoRA?
- Reduce drásticamente la memoria: solo se entrenan los adaptadores,
  no los 135M de parámetros del modelo base.
- El modelo base congelado sirve como referencia (π_ref) para la
  penalización KL, sin necesidad de mantener una copia separada.
- Para obtener los logits de referencia, basta con desactivar
  temporalmente las capas LoRA.

Gradiente de política:
    ∇J(θ) = E[(R - b) · ∇ log π_θ(y|x)]

En la práctica, minimizamos la pérdida:
    L = -E[(R - b) · log π_θ(y|x)] + β · KL(π_θ || π_ref)

    donde:
    - π_θ   : la política (modelo base + adaptador LoRA)
    - π_ref : el modelo de referencia (modelo base SIN adaptador LoRA)
    - β     : coeficiente que controla cuánto puede desviarse la política

Flujo del entrenamiento:
    1. Muestrear un batch de preguntas del dataset.
    2. Para cada pregunta, generar una respuesta con muestreo (sampling).
    3. Calcular la recompensa: ¿acertó la letra correcta?
    4. Forward pass CON adaptador → logits de π_θ
       → log π_θ(y|x) para REINFORCE
    5. Forward pass SIN adaptador → logits de π_ref
       → KL(π_θ || π_ref) para la penalización
    6. Calcular pérdida total = REINFORCE + β · KL.
    7. Backpropagation y actualización de parámetros LoRA.
"""

import os
import json
import random
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


# ============================================================================
# Configuración
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenar un LLM con REINFORCE para MCQA"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Nombre del modelo en HuggingFace Hub",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Número de ejemplos por paso de entrenamiento",
    )
    parser.add_argument(
        "--num_steps", type=int, default=500,
        help="Número total de pasos de entrenamiento",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6,
        help="Learning rate del optimizador",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Temperatura para el muestreo durante la generación (exploración)",
    )

    parser.add_argument(
        "--eval_every", type=int, default=50,
        help="Evaluar en validación cada N pasos",
    )
    parser.add_argument(
        "--eval_samples", type=int, default=200,
        help="Número de muestras del set de validación para evaluar",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria para reproducibilidad",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directorio para guardar el modelo entrenado y métricas",
    )
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0,
        help="Valor máximo de la norma del gradiente (gradient clipping)",
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=0.1,
        help="Coeficiente β de penalización KL. Controla cuánto puede "
             "desviarse la política del modelo original (0 = sin restricción)",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="Rango de las matrices de LoRA (r). Más alto = más capacidad, "
             "más parámetros entrenables",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help="Factor de escala de LoRA (α). La contribución del adaptador "
             "se escala por α/r",
    )
    parser.add_argument(
        "--force_cpu", action="store_true",
        help="Forzar el uso de la CPU incluso si CUDA está disponible",
    )
    return parser.parse_args()


# ============================================================================
# Formato del prompt
# ============================================================================

def format_mcqa_prompt(example, tokenizer):
    """
    Convierte un ejemplo de CommonsenseQA en un prompt de chat.

    CommonsenseQA tiene la siguiente estructura:
      - question: str           (texto de la pregunta)
      - choices: {
            "label": ["A", "B", "C", "D", "E"],
            "text":  ["opción1", "opción2", ...]
        }
      - answerKey: str          ("A", "B", "C", "D", o "E")

    Usamos el formato de chat del modelo (chat template) para que
    el modelo siga su formato de instrucciones nativo.
    """
    question = example["question"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]

    # Construir las opciones con formato legible
    options = "\n".join(f"  {label}) {text}" for label, text in zip(labels, texts))

    content = (
        f"Answer the following multiple-choice question. "
        f"Reply with ONLY the letter of the correct answer.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer:"
    )

    # Formatear como mensaje de chat
    messages = [{"role": "user", "content": content}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


# ============================================================================
# Extracción de respuesta y función de recompensa
# ============================================================================

def extract_answer(text):
    """
    Extrae la letra de respuesta (A-E) del texto generado por el modelo.

    Solo se acepta la letra exacta (tras eliminar espacios en blanco).
    Si el modelo no responde exactamente con una letra válida,
    se considera incorrecto. Esto fuerza al modelo a aprender
    el formato de salida preciso.
    """
    text = text.strip().upper()
    if text in ("A", "B", "C", "D", "E"):
        return text
    return ""


def compute_reward(predicted, correct):
    """
    Recompensa basada en reglas (rule-based reward).

    Es la función de recompensa más simple posible:
      - 1.0 si la respuesta predicha coincide con la correcta
      - 0.0 en caso contrario

    Al usar una regla simple, NO necesitamos entrenar un modelo de
    recompensa separado (reward model), que es uno de los componentes
    más costosos en métodos como RLHF.
    """
    return 1.0 if predicted == correct else 0.0


# ============================================================================
# Generación y cálculo de log-probabilidades
# ============================================================================

@torch.no_grad()
def generate_response(model, input_ids, attention_mask, temperature):
    """
    Genera UN solo token de respuesta usando muestreo con temperatura.

    Se ejecuta SIN gradientes (torch.no_grad) porque en REINFORCE
    tratamos el token generado como una muestra fija de la política.
    Los gradientes se calculan después, en compute_log_prob().

    Args:
        model: El modelo de lenguaje (política π_θ)
        input_ids: Tokens del prompt, shape (1, L)
        attention_mask: Máscara de atención, shape (1, L)
        temperature: Controla la aleatoriedad (más alto = más exploración)

    Returns:
        output_ids: Tensor con [prompt + token_respuesta], shape (1, L+1)
    """
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,         # Solo generamos 1 token (la letra)
        do_sample=True,           # Muestreo (necesario para exploración)
        temperature=temperature,  # Controla la entropía del muestreo
        pad_token_id=model.config.eos_token_id,
    )
    return output_ids


def compute_log_prob_and_kl(model, full_ids, prompt_length):
    """
    Calcula log π_θ(y|x) y KL(π_θ || π_ref) usando un SOLO modelo con LoRA.

    En vez de mantener dos modelos separados en memoria, aprovechamos
    que LoRA permite activar/desactivar el adaptador dinámicamente:

      - Adaptador ACTIVADO  → forward pass de π_θ (política actual)
      - Adaptador DESACTIVADO → forward pass de π_ref (modelo original)

    Esto ahorra ~50% de memoria frente a hacer deepcopy del modelo.

    Args:
        model: Modelo con LoRA (PeftModel). El mismo objeto sirve como
               π_θ (con adaptador) y π_ref (sin adaptador).
        full_ids: [prompt_tokens + token_respuesta], shape (1, L+1)
        prompt_length: Longitud del prompt (L)

    Returns:
        log_prob: Escalar (con gradientes) = log π_θ(y|x)
        kl: Escalar (con gradientes) = KL(π_θ || π_ref)
    """
    # === Forward pass CON adaptador LoRA (π_θ) ===
    outputs = model(input_ids=full_ids)
    logits = outputs.logits  # shape: (1, L+1, vocab_size)

    # Logits en la posición L-1 predicen el token en posición L
    next_token_logits = logits[:, prompt_length - 1, :]  # shape: (1, vocab_size)

    # Log-probabilidades normalizadas (reutilizadas para ambos cálculos)
    current_logprobs = F.log_softmax(next_token_logits, dim=-1)  # shape: (1, vocab_size)

    # --- 1) log π_θ(y|x): seleccionar la log-prob del token generado ---
    generated_token_id = full_ids[:, prompt_length]  # shape: (1,)
    log_prob = current_logprobs[0, generated_token_id[0]]  # escalar

    # === Forward pass SIN adaptador LoRA (π_ref) ===
    # Desactivamos temporalmente las capas LoRA para obtener los logits
    # del modelo base congelado. Esto es equivalente a tener un ref_model
    # separado, pero sin gastar memoria adicional.
    with torch.no_grad():
        model.disable_adapter_layers()
        ref_outputs = model(input_ids=full_ids)
        model.enable_adapter_layers()

        ref_logits = ref_outputs.logits
        ref_logprobs = F.log_softmax(
            ref_logits[:, prompt_length - 1, :], dim=-1
        )  # shape: (1, vocab_size)

    # KL(π_θ || π_ref) = Σ_a π_θ(a) · [log π_θ(a) - log π_ref(a)]
    kl = F.kl_div(
        ref_logprobs,       # input  (log-probs de referencia)
        current_logprobs,   # target (log-probs del modelo actual)
        log_target=True,    # ambos argumentos son log-probabilidades
        reduction="batchmean",
    )

    return log_prob, kl


# ============================================================================
# Evaluación
# ============================================================================

@torch.no_grad()
def evaluate(model, tokenizer, dataset, device, num_samples=200):
    """
    Evalúa la precisión del modelo en un subconjunto del dataset.

    Usa decodificación greedy (do_sample=False) para obtener la
    respuesta más probable del modelo (1 solo token).
    """
    model.eval()

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    correct = 0
    total = 0

    for idx in indices:
        example = dataset[idx]
        prompt = format_mcqa_prompt(example, tokenizer)
        correct_answer = example["answerKey"]

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,  # Solo 1 token: la letra de respuesta
            do_sample=False,   # Greedy: respuesta determinista para evaluación
            pad_token_id=model.config.eos_token_id,
        )

        generated_token = output_ids[:, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_token[0], skip_special_tokens=True)
        predicted = extract_answer(generated_text)

        if predicted == correct_answer:
            correct += 1
        total += 1

    model.train()
    accuracy = correct / total if total > 0 else 0
    return accuracy


# ============================================================================
# Bucle de entrenamiento REINFORCE
# ============================================================================

def train():
    args = parse_args()

    # Reproducibilidad
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Dispositivo
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🖥️  Dispositivo: {device}")
    if device.type == "cuda":
        try:
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception as e:
            print(f"    ⚠️  Error al leer propiedades de la GPU: {e}")
            print("    Sugerencia: Si ves errores de compatibilidad, usa --force_cpu")

    # ----------------------------------------------------------------
    # Cargar modelo y tokenizer
    # ----------------------------------------------------------------
    print(f"\n📦 Cargando modelo: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
    )
    model.to(device)

    # ----------------------------------------------------------------
    # Aplicar LoRA
    # ----------------------------------------------------------------
    # LoRA (Low-Rank Adaptation) inyecta matrices de bajo rango A y B
    # en las capas de atención del modelo:
    #
    #   W' = W + (B @ A) · (α / r)
    #
    # donde:
    #   W  = pesos originales (CONGELADOS, sirven como π_ref)
    #   A  = matriz de rango r, inicializada aleatoriamente
    #   B  = matriz de rango r, inicializada a cero
    #   r  = rango de LoRA (controla la capacidad del adaptador)
    #   α  = factor de escala (controla la magnitud de la adaptación)
    #
    # Ventajas para RL:
    #   - Solo se entrenan A y B (≪ parámetros que el modelo completo)
    #   - El modelo base congelado es automáticamente π_ref para KL
    #   - Para obtener logits de π_ref, basta con desactivar el adaptador
    #   - ~50% menos memoria que mantener una copia deepcopy del modelo
    print(f"\n🔧 Aplicando LoRA (r={args.lora_rank}, α={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Capas de atención
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    # Mostrar estadísticas de parámetros
    model.print_trainable_parameters()
    # Internamente esto muestra:
    #   trainable params: X  ||  all params: Y  ||  trainable%: Z%

    # Asegurar que existe el pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------------------------------------------
    # Cargar dataset
    # ----------------------------------------------------------------
    print("\n📚 Cargando dataset CommonsenseQA...")
    dataset = load_dataset("commonsense_qa")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    print(f"    Train:      {len(train_data):,} ejemplos")
    print(f"    Validation: {len(val_data):,} ejemplos")

    # Mostrar un ejemplo formateado
    sample_prompt = format_mcqa_prompt(train_data[0], tokenizer)
    print(f"\n📝 Ejemplo de prompt:\n{'-'*40}")
    print(sample_prompt[:500])
    print(f"{'-'*40}")

    # ----------------------------------------------------------------
    # Optimizador (solo parámetros LoRA)
    # ----------------------------------------------------------------
    # filter(requires_grad) selecciona automáticamente solo los
    # parámetros del adaptador LoRA, que son los únicos entrenables.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_log = []

    # ----------------------------------------------------------------
    # Evaluación inicial (antes de entrenar)
    # ----------------------------------------------------------------
    print("\n📊 Evaluación INICIAL (antes de entrenar)...")
    initial_acc = evaluate(
        model, tokenizer, val_data, device,
        num_samples=args.eval_samples,
    )
    print(f"    Precisión inicial: {initial_acc:.2%}")

    # ----------------------------------------------------------------
    # Entrenamiento
    # ----------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  🚀 REINFORCE - Entrenamiento")
    print(f"{'='*65}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Num steps:     {args.num_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Grad clip:     {args.gradient_clip}")
    print(f"  KL coeff (β):  {args.kl_coeff}")
    print(f"  LoRA rank (r): {args.lora_rank}")
    print(f"  LoRA alpha:    {args.lora_alpha}")
    print(f"{'='*65}\n")

    best_accuracy = initial_acc

    for step in range(1, args.num_steps + 1):
        model.train()

        # Muestrear un batch aleatorio de ejemplos del dataset
        indices = random.sample(range(len(train_data)), args.batch_size)

        batch_log_probs = []
        batch_rewards = []
        batch_kl_penalties = []

        for idx in indices:
            example = train_data[idx]
            prompt = format_mcqa_prompt(example, tokenizer)
            correct_answer = example["answerKey"]

            # Tokenizar el prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            prompt_length = input_ids.shape[1]

            # --------------------------------------------------------
            # PASO 1: Generar respuesta (SIN gradientes)
            # --------------------------------------------------------
            # El modelo (la política π_θ) genera tokens como "acciones".
            # Usamos muestreo con temperatura para explorar diferentes
            # respuestas. Sin exploración, el modelo se quedaría siempre
            # con la misma respuesta.
            full_ids = generate_response(
                model, input_ids, attention_mask,
                temperature=args.temperature,
            )

            # Decodificar el token generado a texto
            generated_token = full_ids[:, prompt_length:]
            generated_text = tokenizer.decode(
                generated_token[0], skip_special_tokens=True
            )
            predicted = extract_answer(generated_text)

            # --------------------------------------------------------
            # PASO 2: Calcular recompensa
            # --------------------------------------------------------
            # Recompensa binaria basada en reglas:
            #   R = 1.0 si predicted == correct_answer
            #   R = 0.0 en caso contrario
            reward = compute_reward(predicted, correct_answer)

            # --------------------------------------------------------
            # PASO 3: Calcular log π_θ(y|x) y KL(π_θ || π_ref)
            # --------------------------------------------------------
            # Usamos el MISMO modelo para ambos cálculos:
            #   - CON adaptador LoRA → logits de π_θ (política actual)
            #   - SIN adaptador LoRA → logits de π_ref (modelo original)
            # No necesitamos una copia separada del modelo.
            log_prob, kl_penalty = compute_log_prob_and_kl(
                model, full_ids, prompt_length,
            )

            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)
            batch_kl_penalties.append(kl_penalty)

        # Si no se procesó ningún ejemplo, saltar
        if len(batch_log_probs) == 0:
            continue

        # ============================================================
        # PASO 4: Calcular pérdida REINFORCE
        # ============================================================
        #
        #   L = -E[(R - b) · log π_θ(y|x)]
        #
        # Componentes:
        #   - R: la recompensa de cada ejemplo
        #   - b: la línea base (baseline), que es la media de R en el batch
        #   - log π_θ(y|x): la log-probabilidad de la respuesta generada
        #
        # ¿Por qué restamos la baseline?
        #   Sin baseline, REINFORCE tiene varianza muy alta. La baseline
        #   centra las ventajas (advantages) alrededor de 0:
        #   - Si R > b: la respuesta fue mejor que la media → reforzar
        #   - Si R < b: la respuesta fue peor que la media → desincentivar
        #   - Si R = b: no hay señal de aprendizaje
        #
        # ¿Por qué el signo negativo?
        #   Queremos MAXIMIZAR la recompensa esperada, pero los
        #   optimizadores de PyTorch MINIMIZAN. El negativo convierte
        #   maximización en minimización.
        # ============================================================

        log_probs_tensor = torch.stack(batch_log_probs)
        rewards_tensor = torch.tensor(batch_rewards, device=device, dtype=torch.float32)

        # Línea base: media de las recompensas del batch
        baseline = rewards_tensor.mean()

        # Ventaja (advantage): cuánto mejor/peor que la media fue cada respuesta
        advantages = rewards_tensor - baseline

        # Loss de REINFORCE
        # .detach() en advantages es por claridad: asegura que los gradientes
        # solo fluyan a través de log_probs_tensor, no de las advantages
        reinforce_loss = -(advantages.detach() * log_probs_tensor).mean()

        # Penalización KL
        # ============================================================
        # L_KL = β · mean(KL(π_θ || π_ref))
        #
        # Esta penalización actúa como regularización: sin ella, el modelo
        # podría optimizar la recompensa colapsando a una distribución
        # degenerada (siempre la misma respuesta). Con la penalización,
        # el modelo debe encontrar un equilibrio entre:
        #   - Maximizar la recompensa (responder correctamente)
        #   - Mantenerse cerca del modelo original (no "olvidar")
        #
        # β alto → más conservador, menos desviación del original
        # β bajo → más agresivo, optimiza más por recompensa
        # β = 0  → sin penalización, REINFORCE puro
        # ============================================================
        kl_penalties_tensor = torch.stack(batch_kl_penalties)
        mean_kl = kl_penalties_tensor.mean()
        kl_loss = args.kl_coeff * mean_kl

        # Loss total = REINFORCE + penalización KL
        loss = reinforce_loss + kl_loss

        # ============================================================
        # PASO 5: Actualizar los parámetros del modelo
        # ============================================================

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: limita la norma del gradiente para evitar
        # actualizaciones demasiado grandes que desestabilicen el entrenamiento
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )

        optimizer.step()

        # ============================================================
        # Logging
        # ============================================================
        # Con recompensa binaria (0/1), la media es directamente la precisión
        accuracy = rewards_tensor.mean().item()

        metrics = {
            "step": step,
            "loss": loss.item(),
            "reinforce_loss": reinforce_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_kl": mean_kl.item(),
            "batch_accuracy": accuracy,
            "baseline": baseline.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }
        metrics_log.append(metrics)

        # Imprimir cada 10 pasos (y el primero)
        if step % 10 == 0 or step == 1:
            print(
                f"  Step {step:4d}/{args.num_steps} │ "
                f"Loss: {metrics['loss']:+8.4f} │ "
                f"RL: {metrics['reinforce_loss']:+7.4f} │ "
                f"KL: {metrics['mean_kl']:.4f} │ "
                f"Acc: {accuracy:5.0%} │ "
                f"‖∇‖: {metrics['grad_norm']:.4f}"
            )

        # ============================================================
        # Evaluación periódica
        # ============================================================
        if step % args.eval_every == 0:
            print(f"\n  📊 Evaluación en paso {step}...")
            val_acc = evaluate(
                model, tokenizer, val_data, device,
                num_samples=args.eval_samples,
            )
            print(f"     Precisión en validación: {val_acc:.2%}")

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                save_path = os.path.join(args.output_dir, "best_adapter")
                model.save_pretrained(save_path)  # Solo guarda el adaptador LoRA
                tokenizer.save_pretrained(save_path)
                print(f"     ✅ Mejor modelo guardado en {save_path}")

            print()

    # ----------------------------------------------------------------
    # Resultados finales
    # ----------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  📊 RESULTADOS FINALES")
    print(f"{'='*65}")

    final_acc = evaluate(
        model, tokenizer, val_data, device,
        num_samples=args.eval_samples,
    )

    print(f"  Precisión INICIAL (antes de entrenar):  {initial_acc:.2%}")
    print(f"  Mejor precisión durante entrenamiento:  {best_accuracy:.2%}")
    print(f"  Precisión FINAL (último checkpoint):    {final_acc:.2%}")
    print(f"{'='*65}")

    # Guardar adaptador LoRA final
    # Solo se guardan los pesos del adaptador (~KB), no el modelo completo (~MB).
    # Para inferencia, se carga el modelo base + adaptador:
    #   model = AutoModelForCausalLM.from_pretrained("SmolLM2-135M-Instruct")
    #   model = PeftModel.from_pretrained(model, "./output/final_adapter")
    final_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n  💾 Adaptador LoRA final guardado en: {final_path}")

    # Guardar métricas
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"  📈 Métricas guardadas en:   {metrics_path}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    train()
