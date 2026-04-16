#!/usr/bin/env python3
"""
REINFORCE + LoRA para reducción del sesgo posicional en MCQA
=============================================================
 
Variante del algoritmo REINFORCE (Williams, 1992) con LoRA donde la señal
de aprendizaje NO es la accuracy sino la CONSISTENCIA POSICIONAL del modelo:
¿responde la misma opción semántica independientemente del orden en que
se presentan las opciones?
 
¿Por qué consistencia en lugar de accuracy?
 
Los LLMs presentan sesgo posicional en MCQA: tienden a elegir ciertas
posiciones (A, B, C...) con mayor probabilidad a priori, independientemente
del contenido semántico de las opciones. Esto no se corrige entrenando con
accuracy sobre un dataset fijo, porque el modelo puede aprender a acertar
memorizando qué posición suele ser la correcta en ese dataset concreto.
 
La recompensa de consistencia ataca directamente ese sesgo: el modelo solo
recibe señal positiva cuando elige la MISMA respuesta semántica (e.g. "Madrid")
sin importar si aparece en la posición A, B o C.
 
Diseño del experimento:
 
- Para cada pregunta se seleccionan 3 opciones: la correcta + 2 incorrectas
  aleatorias. Con 3 opciones hay 3! = 6 permutaciones, manejable en memoria.
- En cada permutación el modelo genera 1 token (la letra elegida).
- La letra se mapea al texto semántico de la opción (e.g. A → "Madrid").
- Recompensa = fracción de permutaciones donde el modelo eligió la moda
  semántica: reward = count(moda) / num_permutations ∈ [1/6, 1.0]
- El gradiente REINFORCE se aplica sobre la media de log-probs de todas
  las permutaciones del mismo ejemplo, con la ventaja (R - baseline) habitual.
 
Formulación matemática:
 
  Para cada ejemplo x con permutaciones {p_1, ..., p_K}:
    - El modelo genera respuesta y_k en la permutación p_k
    - La respuesta semántica s_k = text(option elegida en p_k)
    - moda = argmax_{s} count(s_k == s)
    - R(x) = count(s_k == moda) / K          (consistencia, ∈ [1/K, 1])
 
  Loss total:
    L = -E_x [(R(x) - b) · mean_k[log π_θ(y_k | p_k(x))]] + β · KL(π_θ || π_ref)
 
  donde b es la baseline (media de R en el batch).
"""

import os
import json
import random
import argparse
from itertools import permutations
from datetime import datetime
from collections import Counter
 
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
        description="REINFORCE con recompensa de consistencia posicional para MCQA"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Nombre del modelo en HuggingFace Hub",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help=(
            "Ejemplos por paso. NOTA: cada ejemplo genera 6 permutaciones, "
            "así que el coste efectivo es batch_size × 6 generaciones."
        ),
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
        help="Temperatura para el muestreo durante la generación",
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
        "--output_dir", type=str, default="./output_consistency",
        help="Directorio para guardar el modelo entrenado y métricas",
    )
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0,
        help="Valor máximo de la norma del gradiente",
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=0.1,
        help=(
            "Coeficiente β de penalización KL. Controla cuánto puede "
            "desviarse la política del modelo original (0 = sin restricción)"
        ),
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="Rango de las matrices de LoRA (r)",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help="Factor de escala de LoRA (α)",
    )
    parser.add_argument(
        "--force_cpu", action="store_true",
        help="Forzar el uso de la CPU incluso si CUDA está disponible",
    )
    return parser.parse_args()

# ============================================================================
# Preparación de opciones y permutaciones
# ============================================================================
 
def sample_3_options(example, rng=None):
    """
    Selecciona 3 opciones de CommonsenseQA: la correcta + 2 incorrectas
    aleatorias.
 
    CommonsenseQA tiene 5 opciones (A-E). Reducimos a 3 para que el número
    de permutaciones sea manejable (3! = 6 en lugar de 5! = 120).
 
    Args:
        example: Ejemplo de CommonsenseQA con campos:
            - choices: {"label": [...], "text": [...]}
            - answerKey: str ("A", "B", "C", "D" o "E")
        rng: instancia de random.Random para reproducibilidad (opcional)
 
    Returns:
        selected_texts: Lista de 3 strings con los textos de las opciones.
        correct_text:   String con el texto de la opción correcta.
                        (siempre está incluida en selected_texts)
    """
    if rng is None:
        rng = random
 
    labels = example["choices"]["label"]
    texts  = example["choices"]["text"]
    answer_key = example["answerKey"]
 
    # Separar la correcta de las incorrectas
    correct_text = None
    incorrect_texts = []
    for label, text in zip(labels, texts):
        if label == answer_key:
            correct_text = text
        else:
            incorrect_texts.append(text)
 
    # Muestrear 2 incorrectas aleatorias
    chosen_incorrect = rng.sample(incorrect_texts, 2)
 
    selected_texts = [correct_text] + chosen_incorrect
    # No mezclar aquí: las permutaciones se generan después
    return selected_texts, correct_text

def generate_all_permutations(option_texts):
    """
    Genera todas las permutaciones de una lista de textos de opciones.
 
    Para 3 opciones devuelve 3! = 6 permutaciones.
    Cada permutación es una lista de textos en un orden diferente.
 
    Args:
        option_texts: Lista de K strings (textos de las opciones)
 
    Returns:
        Lista de K! listas, cada una siendo una permutación de option_texts
    """
    return [list(perm) for perm in permutations(option_texts)]

# ============================================================================
# Formato del prompt
# ============================================================================
 
def format_mcqa_prompt_3opts(question, option_texts, tokenizer):
    """
    Construye el prompt de chat para una pregunta con 3 opciones en un orden
    concreto.
 
    A diferencia del original, recibe directamente los TEXTOS de las opciones
    (no el ejemplo entero) para poder reutilizarse con cualquier permutación.
 
    Las letras A/B/C se asignan según el orden recibido, de modo que el mismo
    texto puede aparecer como A en una permutación y como C en otra.
 
    Args:
        question:      String con el texto de la pregunta.
        option_texts:  Lista de 3 strings con los textos en el orden deseado.
        tokenizer:     Tokenizer del modelo (para apply_chat_template).
 
    Returns:
        prompt: String formateado con el template de chat del modelo.
    """
    labels = ["A", "B", "C"]
    options = "\n".join(
        f"  {label}) {text}"
        for label, text in zip(labels, option_texts)
    )
 
    content = (
        "Answer the following multiple-choice question. "
        "Reply with ONLY the letter of the correct answer.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options}\n\n"
        "Answer:"
    )
 
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

# ============================================================================
# Extracción de respuesta
# ============================================================================
 
def extract_answer_3opts(text):
    """
    Extrae la letra de respuesta (A, B o C) del texto generado.
 
    Solo acepta las letras exactas (A/B/C, insensible a mayúsculas y
    espacios). Cualquier otro output se trata como respuesta inválida ("").
    """
    text = text.strip().upper()
    if text in ("A", "B", "C"):
        return text
    return ""

# ============================================================================
# Recompensa de consistencia
# ============================================================================
 
def compute_consistency_reward(responses, option_texts_per_perm):
    """
    Calcula la recompensa de consistencia semántica.
 
    Para cada permutación, el modelo elige una letra (A/B/C). Esa letra se
    traduce al TEXTO de la opción correspondiente en esa permutación. Si el
    modelo tiene sesgo posicional cero, siempre debería elegir el mismo texto
    semántico con independencia de en qué posición aparezca.
 
    La recompensa es la fracción de permutaciones donde el modelo eligió la
    respuesta semántica más votada (moda):
 
        reward = count(moda_semántica) / num_permutations
 
    Rango: [1/K, 1.0]  (K = número de permutaciones)
      - 1/K  → el modelo elige siempre respuestas diferentes (máximo sesgo)
      - 1.0  → el modelo elige siempre el mismo texto (consistencia perfecta)
 
    Args:
        responses:              Lista de K strings (letra elegida o "" si inválida).
                                Ejemplo: ["A", "C", "B", "A", "", "C"]
        option_texts_per_perm:  Lista de K listas de 3 strings.
                                option_texts_per_perm[k][i] es el texto de la
                                opción en la posición i de la permutación k.
 
    Returns:
        reward:       Float en [0, 1]. Recompensa de consistencia.
        modal_text:   String con el texto semántico más votado (o None si
                      todas las respuestas fueron inválidas).
        semantic_answers: Lista de K strings con los textos elegidos
                          (None si la respuesta fue inválida).
    """
    label_to_idx = {"A": 0, "B": 1, "C": 2}
 
    semantic_answers = []
    for letter, option_texts in zip(responses, option_texts_per_perm):
        if letter in label_to_idx:
            idx = label_to_idx[letter]
            semantic_answers.append(option_texts[idx])
        else:
            # Respuesta inválida: no cuenta para la moda
            semantic_answers.append(None)
 
    valid_answers = [s for s in semantic_answers if s is not None]
 
    if not valid_answers:
        # El modelo no dio ninguna respuesta válida: recompensa mínima
        return 0.0, None, semantic_answers
 
    # Moda semántica: el texto elegido con más frecuencia
    counter = Counter(valid_answers)
    modal_text, modal_count = counter.most_common(1)[0]
 
    # Recompensa = fracción de permutaciones (incluyendo inválidas)
    reward = modal_count / len(responses)
 
    return reward, modal_text, semantic_answers


# ============================================================================
# Generación y cálculo de log-probabilidades
# ============================================================================
 
@torch.no_grad()
def generate_response(model, input_ids, attention_mask, temperature):
    """
    Genera UN solo token de respuesta usando muestreo con temperatura.
 
    Idéntico al original: sin gradientes, 1 token, do_sample=True.
    """
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=True,
        temperature=temperature,
        pad_token_id=model.config.eos_token_id,
    )
    return output_ids


def compute_log_prob_and_kl(model, full_ids, prompt_length):
    """
    Calcula log π_θ(y|x) y KL(π_θ || π_ref) para UNA generación.
 
    Aprovecha que LoRA permite activar/desactivar el adaptador para obtener
    logits de π_θ (con adaptador) y de π_ref (sin adaptador) con el mismo
    objeto modelo, sin duplicar memoria.
 
    Args:
        model:         PeftModel con LoRA.
        full_ids:      [prompt_tokens + token_respuesta], shape (1, L+1)
        prompt_length: Longitud L del prompt.
 
    Returns:
        log_prob: Escalar con gradientes = log π_θ(y|x)
        kl:       Escalar con gradientes = KL(π_θ || π_ref)
    """
    # Forward CON adaptador → π_θ
    outputs = model(input_ids=full_ids)
    logits = outputs.logits                         # (1, L+1, vocab)
    next_token_logits = logits[:, prompt_length - 1, :]
    current_logprobs = F.log_softmax(next_token_logits, dim=-1)
 
    generated_token_id = full_ids[:, prompt_length]
    log_prob = current_logprobs[0, generated_token_id[0]]
 
    # Forward SIN adaptador → π_ref
    with torch.no_grad():
        model.disable_adapter_layers()
        ref_outputs = model(input_ids=full_ids)
        model.enable_adapter_layers()
 
    ref_logits = ref_outputs.logits
    ref_logprobs = F.log_softmax(
        ref_logits[:, prompt_length - 1, :], dim=-1
    )
 
    kl = F.kl_div(
        ref_logprobs,
        current_logprobs,
        log_target=True,
        reduction="batchmean",
    )
 
    return log_prob, kl


# ============================================================================
# Evaluación
# ============================================================================
 
@torch.no_grad()
def evaluate(model, tokenizer, dataset, device, num_samples=200):
    """
    Evalúa el modelo en dos dimensiones sobre un subconjunto del dataset:
 
    1. Consistency score: recompensa de consistencia media (métrica principal,
       la que estamos optimizando).
 
    2. Modal accuracy: fracción de ejemplos donde la respuesta modal del modelo
       (la que elige con más frecuencia entre permutaciones) coincide con la
       respuesta correcta. Métrica secundaria de monitoreo.
 
    Usa decodificación greedy (do_sample=False) para evaluación determinista.
    """
    model.eval()
 
    # Fijar semilla para muestrear siempre las mismas 2 incorrectas en eval
    eval_rng = random.Random(0)
 
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
 
    total_consistency = 0.0
    total_modal_correct = 0
    total = 0
 
    for idx in indices:
        example = dataset[idx]
        question = example["question"]
        correct_answer = example["answerKey"]
 
        # Seleccionar 3 opciones (correcta + 2 incorrectas)
        selected_texts, correct_text = sample_3_options(example, rng=eval_rng)
 
        # Generar todas las permutaciones
        all_perms = generate_all_permutations(selected_texts)
 
        responses = []
        for perm_texts in all_perms:
            prompt = format_mcqa_prompt_3opts(question, perm_texts, tokenizer)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
 
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,           # Greedy para evaluación
                pad_token_id=model.config.eos_token_id,
            )
            generated_token = output_ids[:, input_ids.shape[1]:]
            generated_text = tokenizer.decode(
                generated_token[0], skip_special_tokens=True
            )
            responses.append(extract_answer_3opts(generated_text))
 
        # Calcular consistencia
        consistency, modal_text, _ = compute_consistency_reward(
            responses, all_perms
        )
        total_consistency += consistency
 
        # Comprobar si la moda coincide con la respuesta correcta
        if modal_text == correct_text:
            total_modal_correct += 1
 
        total += 1
 
    model.train()
 
    mean_consistency = total_consistency / total if total > 0 else 0.0
    modal_accuracy   = total_modal_correct / total if total > 0 else 0.0
    return mean_consistency, modal_accuracy


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
 
    # RNG separado para muestrear las 2 incorrectas durante training
    # (independiente del RNG global para no contaminar otros sorteos)
    train_rng = random.Random(args.seed + 1)
 
    # Dispositivo
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    print(f"🖥️  Dispositivo: {device}")
    if device.type == "cuda":
        try:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception as e:
            print(f"   ⚠️  Error al leer propiedades de la GPU: {e}")
 
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
    print(f"\n🔧 Aplicando LoRA (r={args.lora_rank}, α={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()
 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    # ----------------------------------------------------------------
    # Cargar dataset
    # ----------------------------------------------------------------
    print("\n📚 Cargando dataset CommonsenseQA...")
    dataset   = load_dataset("commonsense_qa")
    train_data = dataset["train"]
    val_data   = dataset["validation"]
    print(f"   Train: {len(train_data):,} ejemplos")
    print(f"   Validation: {len(val_data):,} ejemplos")
 
    # Mostrar un ejemplo de cómo se construyen las permutaciones
    sample_ex = train_data[0]
    sample_texts, sample_correct = sample_3_options(sample_ex, rng=train_rng)
    sample_perms = generate_all_permutations(sample_texts)
    print(f"\n📝 Ejemplo de pregunta: {sample_ex['question']}")
    print(f"   3 opciones seleccionadas: {sample_texts}")
    print(f"   Respuesta correcta:       '{sample_correct}'")
    print(f"   Número de permutaciones:  {len(sample_perms)}")
    print(f"\n   Permutación 0 (prompt abreviado):")
    sample_prompt = format_mcqa_prompt_3opts(
        sample_ex["question"], sample_perms[0], tokenizer
    )
    print(f"   {sample_prompt[:300]}...")
    print(f"   {'─'*50}")
 
    # ----------------------------------------------------------------
    # Optimizador
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
 
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_log = []
 
    # ----------------------------------------------------------------
    # Evaluación inicial
    # ----------------------------------------------------------------
    print("\n📊 Evaluación INICIAL (antes de entrenar)...")
    init_consistency, init_modal_acc = evaluate(
        model, tokenizer, val_data, device,
        num_samples=args.eval_samples,
    )
    print(f"   Consistency score:  {init_consistency:.4f}")
    print(f"   Modal accuracy:     {init_modal_acc:.2%}")
    print()
 
    # ----------------------------------------------------------------
    # Cabecera de entrenamiento
    # ----------------------------------------------------------------
    print(f"{'='*70}")
    print(f" 🚀 REINFORCE — Recompensa de Consistencia Posicional")
    print(f"{'='*70}")
    print(f"  Batch size:       {args.batch_size} ejemplos × 6 perms = {args.batch_size*6} generaciones/paso")
    print(f"  Num steps:        {args.num_steps}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  Grad clip:        {args.gradient_clip}")
    print(f"  KL coeff (β):     {args.kl_coeff}")
    print(f"  LoRA rank (r):    {args.lora_rank}")
    print(f"  LoRA alpha (α):   {args.lora_alpha}")
    print(f"{'='*70}\n")
 
    best_consistency = init_consistency
 
    for step in range(1, args.num_steps + 1):
        model.train()
 
        # Muestrear batch aleatorio
        indices = random.sample(range(len(train_data)), args.batch_size)
 
        # Acumuladores para el batch
        example_log_probs   = []   # Una entrada por ejemplo: mean(log_probs de sus 6 perms)
        example_rewards     = []   # Una entrada por ejemplo: su consistency reward
        example_kl_penalties = []  # Una entrada por ejemplo: mean(KL de sus 6 perms)
 
        # Para logging adicional
        all_consistencies = []
        all_modal_correct = []
 
        for idx in indices:
            example    = train_data[idx]
            question   = example["question"]
 
            # Seleccionar 3 opciones (correcta + 2 incorrectas aleatorias)
            # Se re-sortean en cada paso para que el modelo no memorice
            # qué incorrectas van siempre juntas con la correcta.
            selected_texts, correct_text = sample_3_options(
                example, rng=train_rng
            )
 
            # Generar todas las permutaciones de esas 3 opciones
            all_perms = generate_all_permutations(selected_texts)
            # all_perms: lista de 6 listas de 3 textos
 
            # --------------------------------------------------------
            # Para cada permutación:
            #   1. Formatear prompt
            #   2. Generar 1 token (sin gradientes)
            #   3. Registrar respuesta textual para la recompensa
            #   4. Calcular log π_θ(y|x) y KL (con gradientes)
            # --------------------------------------------------------
            perm_log_probs    = []
            perm_kl_penalties = []
            perm_responses    = []   # letras generadas (para la recompensa)
 
            for perm_texts in all_perms:
                prompt = format_mcqa_prompt_3opts(question, perm_texts, tokenizer)
                input_ids = tokenizer.encode(
                    prompt, return_tensors="pt"
                ).to(device)
                attention_mask = torch.ones_like(input_ids)
                prompt_length  = input_ids.shape[1]
 
                # PASO 1: Generar respuesta (sin gradientes)
                full_ids = generate_response(
                    model, input_ids, attention_mask,
                    temperature=args.temperature,
                )
 
                # Decodificar el token generado
                generated_token = full_ids[:, prompt_length:]
                generated_text  = tokenizer.decode(
                    generated_token[0], skip_special_tokens=True
                )
                letter = extract_answer_3opts(generated_text)
                perm_responses.append(letter)
 
                # PASO 2: Calcular log-prob y KL (con gradientes)
                log_prob, kl_penalty = compute_log_prob_and_kl(
                    model, full_ids, prompt_length
                )
                perm_log_probs.append(log_prob)
                perm_kl_penalties.append(kl_penalty)
 
            # --------------------------------------------------------
            # RECOMPENSA DE CONSISTENCIA para este ejemplo
            # --------------------------------------------------------
            # La recompensa es por EJEMPLO (no por permutación individual):
            # cuántas de las 6 permutaciones el modelo respondió lo mismo.
            consistency, modal_text, _ = compute_consistency_reward(
                perm_responses, all_perms
            )
 
            # La señal REINFORCE se aplica a la MEDIA de log-probs de las
            # permutaciones: el modelo aprende a ser consistente en TODAS,
            # no solo en alguna.
            mean_log_prob = torch.stack(perm_log_probs).mean()
            mean_kl       = torch.stack(perm_kl_penalties).mean()
 
            example_log_probs.append(mean_log_prob)
            example_rewards.append(consistency)
            example_kl_penalties.append(mean_kl)
 
            # Logging
            all_consistencies.append(consistency)
            modal_is_correct = (modal_text == correct_text) if modal_text else False
            all_modal_correct.append(float(modal_is_correct))
 
        if len(example_log_probs) == 0:
            continue
 
        # ============================================================
        # PÉRDIDA REINFORCE
        # ============================================================
        #
        # L = -E_x [(R(x) - b) · mean_k[log π_θ(y_k | p_k(x))]]
        #       + β · mean_x[mean_k[KL(π_θ || π_ref | p_k(x))]]
        #
        # - R(x): consistencia del ejemplo x (∈ [1/6, 1])
        # - b: baseline = media de R en el batch (reduce varianza)
        # - mean_k[log π_θ(...)]: media de log-probs sobre las 6 perms
        #
        # Intuición:
        #   Si el modelo fue más consistente que la media del batch:
        #     R(x) > b  →  advantage > 0  →  reforzar esa política
        #   Si fue menos consistente:
        #     R(x) < b  →  advantage < 0  →  desincentivar
        # ============================================================
 
        log_probs_tensor = torch.stack(example_log_probs)
        rewards_tensor   = torch.tensor(
            example_rewards, device=device, dtype=torch.float32
        )
 
        baseline   = rewards_tensor.mean()
        advantages = rewards_tensor - baseline
 
        reinforce_loss = -(advantages.detach() * log_probs_tensor).mean()
 
        # Penalización KL (idéntica al original)
        kl_penalties_tensor = torch.stack(example_kl_penalties)
        mean_kl_batch       = kl_penalties_tensor.mean()
        kl_loss             = args.kl_coeff * mean_kl_batch
 
        loss = reinforce_loss + kl_loss
 
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        optimizer.step()
 
        # ============================================================
        # Logging
        # ============================================================
        mean_consistency_batch = float(sum(all_consistencies) / len(all_consistencies))
        mean_modal_correct     = float(sum(all_modal_correct) / len(all_modal_correct))
 
        metrics = {
            "step":              step,
            "loss":              loss.item(),
            "reinforce_loss":    reinforce_loss.item(),
            "kl_loss":           kl_loss.item(),
            "mean_kl":           mean_kl_batch.item(),
            "consistency_score": mean_consistency_batch,
            "modal_accuracy":    mean_modal_correct,
            "baseline":          baseline.item(),
            "grad_norm":         (
                grad_norm.item()
                if isinstance(grad_norm, torch.Tensor)
                else grad_norm
            ),
        }
        metrics_log.append(metrics)
 
        if step % 10 == 0 or step == 1:
            print(
                f"  Step {step:4d}/{args.num_steps} │ "
                f"Loss: {metrics['loss']:+8.4f} │ "
                f"RL: {metrics['reinforce_loss']:+7.4f} │ "
                f"KL: {metrics['mean_kl']:.4f} │ "
                f"Cons: {mean_consistency_batch:.3f} │ "
                f"ModalAcc: {mean_modal_correct:5.0%} │ "
                f"‖∇‖: {metrics['grad_norm']:.4f}"
            )
 
        # ============================================================
        # Evaluación periódica
        # ============================================================
        if step % args.eval_every == 0:
            print(f"\n  📊 Evaluación en paso {step}...")
            val_consistency, val_modal_acc = evaluate(
                model, tokenizer, val_data, device,
                num_samples=args.eval_samples,
            )
            print(f"     Consistency score (val): {val_consistency:.4f}")
            print(f"     Modal accuracy    (val): {val_modal_acc:.2%}")
 
            if val_consistency > best_consistency:
                best_consistency = val_consistency
                save_path = os.path.join(args.output_dir, "best_adapter")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"     ✅ Mejor modelo guardado en {save_path}")
            print()
 
    # ----------------------------------------------------------------
    # Resultados finales
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  📊 RESULTADOS FINALES")
    print(f"{'='*70}")
 
    final_consistency, final_modal_acc = evaluate(
        model, tokenizer, val_data, device,
        num_samples=args.eval_samples,
    )
 
    print(f"  Consistency score INICIAL:  {init_consistency:.4f}")
    print(f"  Consistency score FINAL:    {final_consistency:.4f}")
    print(f"  Mejor consistency (val):    {best_consistency:.4f}")
    print(f"  Modal accuracy INICIAL:     {init_modal_acc:.2%}")
    print(f"  Modal accuracy FINAL:       {final_modal_acc:.2%}")
    print(f"{'='*70}")
 
    # Guardar adaptador final
    final_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n  💾 Adaptador LoRA final guardado en: {final_path}")
 
    # Guardar métricas
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"  📈 Métricas guardadas en: {metrics_path}")
 
 
# ============================================================================
# Entry point
# ============================================================================
 
if __name__ == "__main__":
    train()