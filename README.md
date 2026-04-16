# 🎯 REINFORCE + LoRA para Language Models — MCQA

Implementación educativa del algoritmo **REINFORCE** (Williams, 1992) con
**LoRA** (Low-Rank Adaptation) para ajuste fino eficiente de un modelo
de lenguaje en tareas de **preguntas de opción múltiple (MCQA)**.

## ¿Por qué REINFORCE?

REINFORCE es el método de Reinforcement Learning más simple que se puede aplicar
a un LLM. Comparado con otros métodos:

| Método     | Necesita Crítico | Necesita Reward Model | Complejidad |
|------------|:----------------:|:---------------------:|:-----------:|
| **REINFORCE** | ❌ No         | ❌ No (reglas)        | ⭐ Baja     |
| GRPO       | ❌ No            | ❌ No (reglas)        | ⭐⭐ Media  |
| PPO        | ✅ Sí            | ✅ Sí                 | ⭐⭐⭐ Alta |
| RLHF (PPO) | ✅ Sí            | ✅ Sí                 | ⭐⭐⭐⭐    |

## Conceptos clave

### El problema como RL

Formulamos la tarea MCQA como un problema de Reinforcement Learning:

- **Política (π_θ)**: El modelo de lenguaje con adaptador LoRA
- **Referencia (π_ref)**: El mismo modelo base, sin adaptador LoRA
- **Estado (s)**: El prompt con la pregunta y las opciones
- **Acción (a)**: Los tokens que genera el modelo como respuesta
- **Recompensa (R)**: 1.0 si la respuesta es correcta, 0.0 si no

### El algoritmo REINFORCE

REINFORCE optimiza directamente la política usando el gradiente:

```
L = -E[(R - b) · log π_θ(y|x)] + β · KL(π_θ || π_ref)
```

Donde:
- `R` es la recompensa obtenida
- `b` es la **baseline** (media de recompensas del batch), que reduce la varianza
- `log π_θ(y|x)` es la log-probabilidad de la respuesta generada
- `β` es el coeficiente de penalización KL
- `KL(π_θ || π_ref)` mide cuánto se desvía el modelo del original

Intuitivamente:
- Si `R > b`: la respuesta fue **mejor que la media** → **reforzar** esa respuesta
  (aumentar su probabilidad)
- Si `R < b`: la respuesta fue **peor que la media** → **desincentivar** esa respuesta
  (reducir su probabilidad)

### Penalización KL

Para evitar que el entrenamiento destruya el conocimiento previo del modelo
("catastrophic forgetting" / "reward hacking"), se incluye una **penalización
KL** que mide cuánto se desvía la distribución actual `π_θ` del modelo original
`π_ref`:

- **β alto** → más conservador, el modelo cambia poco
- **β bajo** → más agresivo, optimiza más por recompensa
- **β = 0** → sin penalización, REINFORCE puro

### LoRA (Low-Rank Adaptation)

En vez de entrenar todos los parámetros del modelo (~135M), LoRA inyecta
matrices de bajo rango en las capas de atención:

```
W' = W + (B @ A) · (α / r)    donde W está congelado, solo A y B se entrenan
```

Ventajas clave para RL:
- 💾 **Ahorro de memoria**: Solo se entrenan ~0.1% de los parámetros
- 🧊 **Referencia gratuita**: El modelo base congelado ES `π_ref`
- 🔀 **Un solo modelo**: Para obtener logits de `π_ref`, basta con desactivar
  el adaptador temporalmente (`model.disable_adapter_layers()`)

### Flujo del entrenamiento

```
┌─────────────────────────────────────────────────┐
│                  Para cada paso:                │
│                                                 │
│  1. Muestrear batch de preguntas del dataset    │
│                    ↓                            │
│  2. Generar respuestas (sampling con temp.)     │
│     [SIN gradientes - son "muestras"]           │
│                    ↓                            │
│  3. Calcular recompensa por reglas              │
│     ¿Acertó la letra? → R=1 / R=0              │
│                    ↓                            │
│  4. Forward CON adaptador LoRA (π_θ)           │
│     → log π_θ(y|x) para REINFORCE              │
│                    ↓                            │
│  5. Forward SIN adaptador LoRA (π_ref)         │
│     → KL(π_θ || π_ref) para penalización       │
│                    ↓                            │
│  6. Loss = REINFORCE + β · KL                  │
│                    ↓                            │
│  7. Backprop + actualizar solo pesos LoRA       │
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
    --batch_size 16 \
    --num_steps 1000 \
    --lr 5e-7 \
    --temperature 0.7 \
    --eval_every 100 \
    --output_dir ./my_experiment
```

### Parámetros principales

| Parámetro         | Default    | Descripción                              |
|-------------------|------------|------------------------------------------|
| `--model_name`    | SmolLM2-135M-Instruct | Modelo base               |
| `--batch_size`    | 8          | Ejemplos por paso                        |
| `--num_steps`     | 500        | Pasos totales de entrenamiento           |
| `--lr`            | 1e-6       | Learning rate                            |
| `--temperature`   | 0.8        | Temperatura de muestreo (exploración)    |
| `--eval_every`    | 50         | Frecuencia de evaluación                 |
| `--eval_samples`  | 200        | Muestras para evaluación                 |
| `--gradient_clip` | 1.0        | Norma máxima del gradiente               |
| `--kl_coeff`      | 0.1        | Coeficiente β de penalización KL         |
| `--lora_rank`     | 16         | Rango de las matrices LoRA (r)           |
| `--lora_alpha`    | 32         | Factor de escala de LoRA (α)             |
| `--seed`          | 42         | Semilla aleatoria                        |

## Resultados esperados

Con la configuración por defecto:

- **Antes del entrenamiento**: ~20-25% de precisión (casi aleatorio para 5 opciones)
- **Después de 500 pasos**: ~30-45% de precisión (mejora significativa)

> ⚠️ SmolLM2-135M es un modelo muy pequeño. No esperes precisiones muy altas,
> pero sí deberías ver una mejora clara sobre la precisión inicial.

## Estructura del proyecto

```
rl-learn/
├── README.md              ← Este archivo
├── requirements.txt       ← Dependencias (torch, transformers, peft, ...)
├── train_reinforce.py     ← Script principal de entrenamiento
└── output/                ← (creado al entrenar)
    ├── best_adapter/      ← Mejor adaptador LoRA (solo pesos del adaptador)
    ├── final_adapter/     ← Adaptador LoRA final
    └── metrics.json       ← Métricas de entrenamiento
```

## Siguientes pasos

Una vez entiendas REINFORCE, puedes explorar métodos más avanzados:

1. **REINFORCE con múltiples muestras**: Generar varias respuestas por pregunta
   para tener una mejor estimación del gradiente.

2. ~~**Penalización KL**~~: ✅ **Implementado** — Se añadió el término
   `β · KL(π_θ || π_ref)` usando el modelo base como referencia.

3. **GRPO** (Group Relative Policy Optimization): Similar a REINFORCE pero usa
   normalización por grupo de las ventajas. Es el método usado por DeepSeek.

4. **PPO** (Proximal Policy Optimization): Añade clipping del ratio de políticas
   y un modelo crítico para estimar la ventaja. Más estable pero más complejo.

5. ~~**LoRA**~~: ✅ **Implementado** — Adaptación de bajo rango con `peft`.
   El modelo base congelado sirve simultáneamente como π_ref para KL.

## Referencias

- Williams, R.J. (1992). *Simple statistical gradient-following algorithms for
  connectionist reinforcement learning*. Machine Learning, 8, 229–256.
- Hu, E.J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
  arXiv:2106.09685.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models*. arXiv:2402.03300. (Introduce GRPO)
