# 🤖 Turing Duel

Simulación de una **"Prueba de Turing inversa"** entre dos modelos de lenguaje (como GPT-4, Claude, LLaMA, Mistral, DeepSeek, etc.), con el objetivo de observar si alguno se delata como modelo o sospecha que el otro no es humano.

## 📌 Descripción

Este proyecto crea un entorno controlado donde dos modelos conversan entre sí por turnos, comenzando con un `prompt inicial`. Se registra toda la conversación y se evalúa en cada turno si alguno:

- Se autodelata (dice explícitamente que es un modelo)
- Sospecha que el interlocutor no es humano

Si ocurre alguno de estos eventos, la conversación se detiene.

## 📂 Estructura

- `turing_duel.ipynb`: Notebook principal con la lógica del duelo, análisis y visualización.
- `conversational_models_sorted.csv`: Lista de modelos de lenguaje disponibles en https://ai.azure.com/explore/models.
- `README.md`: Este archivo.

## 🚀 Requisitos

- Python 3.8+
- OpenAI API Key (para usar GPT)
- Las siguientes bibliotecas:

```bash
pip install openai pandas dash plotly