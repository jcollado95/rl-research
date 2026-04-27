# 🎯 REINFORCE + LoRA para Language Models — MCQA

Implementación educativa del algoritmo **REINFORCE** (Williams, 1992) con
**LoRA** (Low-Rank Adaptation) para ajuste fino eficiente de un modelo
de lenguaje en tareas de **preguntas de opción múltiple (MCQA)**.

## ¿Por qué Consistencia en lugar de Accuracy?

Los LLMs presentan **sesgo posicional** en MCQA: tienden a elegir ciertas posiciones (A, B, C...) con mayor probabilidad a priori, independientemente del contenido semántico de las opciones. Esto no se corrige fácilmente entrenando con accuracy sobre un dataset fijo, porque el modelo puede aprender a acertar memorizando la posición estadísticamente más probable.

La recompensa de **consistencia** ataca directamente ese sesgo: el modelo solo recibe señal positiva cuando elige la MISMA respuesta semántica sin importar si aparece en la posición A, B o C. Para cada pregunta, generamos múltiples permutaciones de las opciones y medimos cuántas coinciden semánticamente.

### El problema como RL

Formulamos la tarea MCQA como un problema de Reinforcement Learning:

- **Política (π_θ)**: El modelo de lenguaje con adaptador LoRA
- **Referencia (π_ref)**: El mismo modelo base, sin adaptador LoRA
- **Estado (s)**: El prompt con la pregunta y un ordenamiento específico de las opciones
- **Acción (a)**: Los tokens que genera el modelo como respuesta (A, B, C)
- **Recompensa (R)**: Rango [1/K, 1.0] basado en la fracción de permutaciones donde el modelo fue consistente con su propia moda semántica.
  > 💡 **Nota matemática (Principio del Palomar / Cajas de Dirichlet)**: 
  > En la práctica, si evaluamos $K=6$ permutaciones para 3 clases semánticas (opciones), el modelo podría tener un sesgo posicional extremo y decir siempre la opción de la primera letra (p.ej. "A"). Si eso ocurre, habrá escogido cada opción semántica exactamente 2 veces. Por el Principio del Palomar (repartir 6 elementos generados en 3 posibles respuestas de texto), al menos un texto debe haber sido elegido $\lceil 6/3 \rceil = 2$ veces. Por tanto, la moda mínima (y recompensa base sin errores de sintaxis) siempre es de **2/6**. 
  > Que el peor caso "válido" no sea 0 sino 2/6 **no afecta a REINFORCE**. Como la pérdida resta la media local del batch (`R - b`), un sesgo continuo producirá una ventaja `R - b = 0`, evitando que se den gradientes positivos para políticas extremadamente sesgadas.

### El algoritmo REINFORCE

REINFORCE optimiza directamente la política usando el gradiente:

```
L = -E[(R - b) · mean_K(log π_θ(y_k|x_k))] + β · mean_K(KL(π_θ || π_ref))
```

Donde:
- `R` es la recompensa de consistencia obtenida para el ejemplo completo
- `b` es la **baseline** (media de recompensas del batch), que reduce la varianza
- `mean_K(log ...)` promedio de log-probs de las respuestas en todas las permutaciones del ejemplo
- `β` es el coeficiente de penalización KL
- `KL(π_θ || π_ref)` mide cuánto se desvía el modelo del original

Intuitivamente:
- Si `R > b`: el modelo fue **más consistente que la media** → **reforzar** las respuestas dadas
- Si `R < b`: el modelo fue **menos consistente** → **desincentivar** esas respuestas

### Penalización KL

Para evitar que el entrenamiento destruya el conocimiento previo del modelo ("catastrophic forgetting" / "reward hacking"), se incluye una penalización KL que mide cuánto se desvía la distribución actual `π_θ` del modelo original `π_ref`.

### LoRA (Low-Rank Adaptation)

En vez de entrenar todos los parámetros del modelo (~135M), LoRA inyecta matrices de bajo rango en las capas de atención, lo que ahorra memoria y permite usar el modelo base congelado como `π_ref` sin duplicar la red en VRAM.

### Flujo del entrenamiento

```
┌─────────────────────────────────────────────────┐
│                  Para cada paso:                │
│                                                 │
│  1. Muestrear batch de preguntas del dataset    │
│                    ↓                            │
│  2. Para cada pregunta, generar K permutaciones │
│     (K=6 en 3 opciones).                        │
│                    ↓                            │
│  3. Generar respuestas (sampling con temp.)     │
│                    ↓                            │
│  4. Calcular la recompensa de consistencia:     │
│     ¿Cuántas de las 6 respuestas semánticas     │
│     coinciden con la respuesta mayoritaria?     │
│                    ↓                            │
│  5. Forward CON adaptador LoRA (π_θ)            │
│     → log π_θ(y|x) para REINFORCE               │
│                    ↓                            │
│  6. Forward SIN adaptador LoRA (π_ref)          │
│     → KL(π_θ || π_ref) para penalización        │
│                    ↓                            │
│  7. Loss = REINFORCE + β · KL                   │
│                    ↓                            │
│  8. Backprop + actualizar solo pesos LoRA       │
└─────────────────────────────────────────────────┘
```

## Setup

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento básico

```bash
python train_reinforce.py
```

### Con parámetros personalizados

```bash
python train_reinforce.py \
    --batch_size 4 \
    --num_steps 500 \
    --lr 1e-6 \
    --temperature 0.8 \
    --eval_every 50 \
    --output_dir ./mi_experimento_consistencia
```

### Parámetros principales

| Parámetro         | Default    | Descripción                              |
|-------------------|------------|------------------------------------------|
| `--model_name`    | SmolLM2-135M-Instruct | Modelo base               |
| `--batch_size`    | 4          | Ejemplos por paso (¡multiplicado por 6 perms!)|
| `--num_steps`     | 500        | Pasos totales de entrenamiento           |
| `--lr`            | 1e-6       | Learning rate                            |
| `--temperature`   | 0.8        | Temperatura de muestreo (exploración)    |
| `--eval_every`    | 50         | Frecuencia de evaluación                 |
| `--eval_samples`  | 200        | Muestras para evaluación                 |
| `--gradient_clip` | 1.0        | Norma máxima del gradiente               |
| `--kl_coeff`      | 0.1        | Coeficiente β de penalización KL         |

## Resultados esperados

La métrica principal optimizada es el `Consistency score`:

- **Antes del entrenamiento**: El modelo presentará cierta variabilidad al rotar las opciones.
- **Después del entrenamiento**: La consistencia debería aumentar significativamente (hacia 1.0), indicando que el modelo es robusto a las permutaciones y no sufre de sesgo posicional.

## Estructura del proyecto

```
rl-research/
├── README.md                 ← Este archivo
├── requirements.txt          ← Dependencias (torch, transformers, peft, matplotlib, ...)
├── train_reinforce.py        ← Script principal de entrenamiento
├── evaluate_models.py        ← Script de evaluación detallada
├── visualize_results.py      ← Visualización de resultados de un experimento
├── compare_experiments.py    ← Comparación de múltiples experimentos
├── train.sbs                 ← Script SLURM para lanzar entrenamiento
└── experiments/              ← (creado automáticamente al entrenar)
    └── 2026-04-23_11-08_bs4_lr5e-06/    ← un experimento
        ├── config.json           ← Args, git hash, timestamp, versiones
        ├── metrics.json          ← Métricas paso a paso (con eval periódico)
        ├── summary.json          ← Resumen: métricas iniciales/finales, wall-clock
        ├── best_adapter/         ← Mejor adaptador LoRA (por consistency)
        ├── final_adapter/        ← Adaptador LoRA final
        ├── eval/                 ← Resultados de evaluación
        │   ├── eval_base.json
        │   └── eval_trained.json
        └── plots/                ← Gráficas (generadas por visualize_results.py)
            ├── training_curves.png
            └── eval_comparison.png
```

## Tracking & Reproducibilidad

Cada ejecución de `train_reinforce.py` crea automáticamente un directorio con
nombre `YYYY-MM-DD_HH-MM_bs{batch_size}_lr{lr}` que contiene:

- **`config.json`**: Todos los argumentos de la CLI, hash de git, versiones de
  Python/PyTorch/CUDA.
- **`metrics.json`**: Métricas por paso. Cuando se ejecuta evaluación periódica,
  las métricas de validación (consistency, modal accuracy, overall accuracy) se
  incluyen en el mismo registro del paso.
- **`summary.json`**: Resumen con métricas iniciales, finales, mejor consistency,
  y tiempo total de entrenamiento.

## Visualización

### Gráficas de un experimento

```bash
# Mostrar gráficas interactivas
python visualize_results.py experiments/2026-04-23_11-08_bs4_lr5e-06/

# Guardar como PNG
python visualize_results.py experiments/2026-04-23_11-08_bs4_lr5e-06/ --save

# Guardar como PDF (para papeles)
python visualize_results.py experiments/2026-04-23_11-08_bs4_lr5e-06/ --save --format pdf
```

Genera:
- **Training curves** (multi-panel): loss, consistency, modal accuracy, KL, grad norm
  con curvas suavizadas (EMA) y datos crudos de fondo
- **Evaluation comparison**: Bar chart (base vs entrenado) + sesgo posicional

### Comparación de múltiples experimentos

```bash
# Tabla comparativa en terminal
python compare_experiments.py experiments/exp_1/ experiments/exp_2/

# Con gráficas superpuestas guardadas
python compare_experiments.py experiments/2026-*/ --save

# Exportar tabla LaTeX para tu paper
python compare_experiments.py experiments/2026-*/ --latex
python compare_experiments.py experiments/2026-*/ --latex --latex-output tabla.tex
```

## Evaluación

```bash
# Evaluar modelo base
python evaluate_models.py --model_name Qwen/Qwen3-0.6B

# Evaluar modelo entrenado (guardando en directorio del experimento)
python evaluate_models.py \
    --model_name Qwen/Qwen3-0.6B \
    --adapter_path experiments/2026-04-23_11-08_bs4_lr5e-06/best_adapter \
    --experiment_dir experiments/2026-04-23_11-08_bs4_lr5e-06/ \
    --output_file eval_trained.json
```

## Siguientes pasos

Una vez entiendas REINFORCE, puedes explorar métodos más avanzados:

1. **REINFORCE con múltiples muestras**: Generar varias respuestas por pregunta
   para tener una mejor estimación del gradiente.

2. ~~**Penalización KL**~~: ✅ **Implementado** — Se añadió el término
   `β · KL(π_θ || π_ref)` usando el modelo base como referencia.

3. **GRPO** (Group Relative Policy Optimization): Similar a REINFORCE pero usa
   normalización por grupo de las ventajas. Es el método usado por DeepSeek.

4. ~~**LoRA**~~: ✅ **Implementado** — Adaptación de bajo rango con `peft`.
   El modelo base congelado sirve simultáneamente como π_ref para KL.

## Referencias

- Williams, R.J. (1992). *Simple statistical gradient-following algorithms for
  connectionist reinforcement learning*. Machine Learning, 8, 229–256.
- Hu, E.J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
  arXiv:2106.09685.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models*. arXiv:2402.03300. (Introduce GRPO)
