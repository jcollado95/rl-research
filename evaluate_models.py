#!/usr/bin/env python3
"""
Script de evaluación para revisar modelos originales y entrenados.
Calcula métricas como consistencia, modal accuracy, perfect accuracy y muestra el
sesgo porcentual hacia las distintas letras (A, B, C) a lo largo de las permutaciones.
"""

import os
import json
import random
import argparse
from tqdm import tqdm
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from peft import PeftModel

# Importamos las funciones base de nuestro script de entrenamiento para mantener la compatibilidad empírica
from train_reinforce import (
    sample_3_options,
    generate_all_permutations,
    format_mcqa_prompt_3opts,
    extract_answer_3opts,
    compute_consistency_reward
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluar modelo: sesgo posicional y consistencia")
    parser.add_argument(
        "--model_name", type=str, 
        default="Qwen/Qwen3-0.6B"
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None, 
        help="Ruta al adaptador LoRA (si no se proporciona se evaluará el modelo original)"
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="commonsense_qa",
        help="Nombre del dataset en HuggingFace Hub o path local",
    )
    parser.add_argument(
        "--split", type=str, default="validation", 
        help="Split del dataset a evaluar (train, validation, test)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, 
        help="Límite máximo de muestras a evaluar. Por defecto evalúa todo el split."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Nombre del archivo de salida. Si no se indica, se autogenera basado en el modelo y el adaptador."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Fijar semillas
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")

    # =========================================================
    # CARGAR MODELO
    # =========================================================
    print(f"📦 Cargando tokenizer y modelo base: {args.model_name}")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=model_dtype)
    
    if args.adapter_path:
        print(f"🔧 Cargando adaptador LoRA desde: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print("💡 Evaluando MODELO ORIGINAL (sin adaptador).")
        
    model.to(device)
    model.eval()

    # =========================================================
    # CARGAR DATASET
    # =========================================================
    dataset = load_dataset("commonsense_qa", split=args.split)
    
    # Manejar split limit
    if args.num_samples and args.num_samples < len(dataset):
        # Muestreo estocástico uniforme de los ejemplos a evaluar
        indices = random.sample(range(len(dataset)), args.num_samples)
        dataset = dataset.select(indices)
    
    print(f"📚 Dataset a evaluar: {len(dataset)} ejemplos (Split: {args.split})")

    # =========================================================
    # EVALUACIÓN
    # =========================================================
    results = []
    
    total_consistency = 0.0
    total_modal_correct = 0
    total_perfect_accuracy = 0
    total_predictions = 0
    
    # Diccionario para trackear sesgo global en posición
    letter_counts = {"A": 0, "B": 0, "C": 0, "INVALID": 0}
    
    # Reproducibilidad paralela que no contamine generadores
    eval_rng = random.Random(args.seed)

    for i, example in enumerate(tqdm(dataset, desc="Evaluando")):
        question = example["question"]
        correct_answer = example["answerKey"]
        
        # En splits como 'test', answerKey a menudo está vacío.
        # En esos casos se puede seguir infiriendo consistencia, pero no su accuracy empírica.
        evaluate_accuracy = bool(correct_answer and correct_answer.strip())

        # Aislar 3 opciones de las 5 del dataset usando la misma lógica de "train_reinforce.py"
        try:
            selected_texts, correct_text = sample_3_options(example, rng=eval_rng)
        except Exception:
            # Si un dataset no cumple la misma estructura de choices, pasamos
            continue
            
        all_perms = generate_all_permutations(selected_texts)

        responses_letters = []
        for perm_texts in all_perms:
            prompt = format_mcqa_prompt_3opts(question, perm_texts, tokenizer)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=False,  # Greedy estricto, no exploración.
                    pad_token_id=(
                        model.config.eos_token_id[0]
                        if isinstance(model.config.eos_token_id, list)
                        else model.config.eos_token_id
                    ),
                )
                
            generated_token = output_ids[:, input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_token[0], skip_special_tokens=True)
            letra = extract_answer_3opts(generated_text)
            
            responses_letters.append(letra)
            
            # Registro de posicionamiento (sesgo bruto)
            if letra in ("A", "B", "C"):
                letter_counts[letra] += 1
            else:
                letter_counts["INVALID"] += 1
            total_predictions += 1

        # Métrica base de consistencia extraída
        consistency, modal_text, semantic_answers = compute_consistency_reward(responses_letters, all_perms)

        is_correct = None
        is_perfect_accuracy = None
        
        # Guardaremos el dict
        example_result = {
            "question": question,
            "correct_text": correct_text if evaluate_accuracy else None,
            "permutations": all_perms,
            "raw_responses_letters": responses_letters,
            "semantic_responses": semantic_answers,
            "modal_answer": modal_text,
            "consistency_score": consistency,
        }

        # Si tenemos un AnswerKey en el dataset (que es lo normal para validation)
        if evaluate_accuracy:
            is_correct = (modal_text == correct_text)
            is_perfectly_consistent = (consistency == 1.0) # Seis repeticiones coherentes
            
            # Perfect Accuracy = Fue perfectamente determinista a la semántica Y además acertó
            is_perfect_accuracy = bool(is_correct and is_perfectly_consistent)
            
            example_result["is_correct"] = is_correct
            example_result["is_perfect_accuracy"] = is_perfect_accuracy
            
            if is_correct:
                total_modal_correct += 1
            if is_perfect_accuracy:
                total_perfect_accuracy += 1

        total_consistency += consistency
        results.append(example_result)

    # =========================================================
    # REPORTE Y MANTENIMIENTO
    # =========================================================
    num_eval = len(results)
    if num_eval == 0:
        print("\n❌ No se evaluaron ejemplos válidos.")
        return

    mean_consistency = total_consistency / num_eval
    
    # Calcular promedios generales descartando vacíos de label
    eval_acc_count = sum(1 for r in results if r.get("is_correct") is not None)
    
    print(f"\n{'='*70}")
    adapter_text = f"(CON '{args.adapter_path}')" if args.adapter_path else "(MODELO BASE ORIGINARIO)"
    print(f" 📊 RESULTADOS DE LA EVALUACIÓN {adapter_text}")
    print(f"{'='*70}")
    print(f"  Consistency Score (Media):   {mean_consistency:.4f}")
    
    if eval_acc_count > 0:
        modal_accuracy = total_modal_correct / eval_acc_count
        perfect_accuracy = total_perfect_accuracy / eval_acc_count
        print(f"  Modal Accuracy:              {modal_accuracy:.2%}")
        print(f"  Perfect Accuracy:            {perfect_accuracy:.2%} (Consistencia absoluta de 6/6 Y Acertó)")
    else:
        modal_accuracy = None
        perfect_accuracy = None
        print("  Modal y Perfect Accuracy no computados (El dataset proporcionado no tiene la meta-respuesta).")
        
    print(f"\n  🎯 SESGO POSICIONAL BRUTO (Agregado de las {total_predictions} respuestas extraídas):")
    for b_let in ["A", "B", "C", "INVALID"]:
        pct = letter_counts[b_let] / max(1, total_predictions)
        print(f"     Opción {b_let:7s}: {pct:6.2%} ({letter_counts[b_let]} veces)")
    print(f"{'='*70}\n")

    # Guardado de output de metricas y detalle de trayectorias
    report_data = {
        "metrics": {
            "mean_consistency": float(mean_consistency),
            "modal_accuracy": float(modal_accuracy) if modal_accuracy is not None else None,
            "perfect_accuracy": float(perfect_accuracy) if perfect_accuracy is not None else None,
            "positional_bias": {k: float(v / max(1, total_predictions)) for k, v in letter_counts.items()}
        },
        "details": results
    }
    
    if args.output_file is not None:
        out_path = args.output_file
        # Asegurar que el directorio parent existe
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    else:
        model_basename = os.path.basename(os.path.normpath(args.model_name))
        if args.adapter_path:
            # Typical adapter path: experiments/2026-04.../best_adapter
            adapter_dir = os.path.dirname(os.path.normpath(args.adapter_path))
            adapter_id = os.path.basename(adapter_dir)
            if not adapter_id or adapter_id == ".":
                adapter_id = os.path.basename(os.path.normpath(args.adapter_path))
                
            out_filename = f"eval_{model_basename}_{adapter_id}.json"
            out_dir = os.path.join(adapter_dir, "eval")
        else:
            out_filename = f"eval_{model_basename}_base.json"
            out_dir = "./eval"
            
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_filename)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
        
    print(f"💾 Se han guardado detalles completos por permutación en: {out_path}")

if __name__ == "__main__":
    main()
