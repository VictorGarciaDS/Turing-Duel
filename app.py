#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys
import random
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc


# In[8]:


# === CONFIGURACIÓN Y DATOS ===
load_dotenv()

OPEN_API_ENDPOINT ="https://models.inference.ai.azure.com"
OPEN_API_KEY = os.getenv("OPEN_API_KEY")


# In[9]:


AVAILABLE_MODELS = pd.read_csv("conversational_models_sorted.csv")["model"].dropna().unique().tolist()

AVAILABLE_MODELS = ['gpt-35-turbo', 'gpt-35-turbo-16k', 'gpt-35-turbo-instruct', 'gpt-4', 'gpt-4-32k', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
                    'gpt-4.5-preview', 'gpt-4o', 'gpt-4o-mini', 'gpt2', 'gpt2', 'gpt2-large', 'gpt2-large', 'gpt2-medium', 'gpt2-medium', 'gpt2-xl',
                    'DeepSeek-R1', 'DeepSeek-R1-Distilled-NPU-Optimized', 'DeepSeek-V3', 'DeepSeek-V3-0324', 'Deepseek-R1-Distill-Llama-8B-NIM-microservice']

DETECTION_KEYWORDS = [
    "as an ai", "as a language model", "i am an ai", "i'm an ai",
    "i am not human", "i was trained", "as a bot", "i am artificial",
    "being an ai", "my training data"
]

SUSPECT_KEYWORDS = [
    "are you an ai", "you sound robotic", "you sound like an ai",
    "you're not human", "you're an ai", "you don't sound human"
]

client = OpenAI(base_url=OPEN_API_ENDPOINT, api_key=OPEN_API_KEY)

def fetch_chat_completion(messages, model_name):
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.2,
            max_tokens=100,
            top_p=0.95
        )
        return response.choices[0].message.model_dump()
    except Exception as e:
        print(f"[ERROR] Al llamar a {model_name}: {e}")
        return {"role": "assistant", "content": f"[ERROR: No se pudo generar respuesta del modelo '{model_name}']"}

def self_disclosure(text):
    return any(k in text.lower() for k in DETECTION_KEYWORDS)

def suspects_other(text):
    return any(k in text.lower() for k in SUSPECT_KEYWORDS)


# In[10]:


# === ESTILO DASH ===
terminal_style = {
    'backgroundColor': '#000000',
    'color': '#00FF00',
    'fontFamily': 'monospace',
    'padding': '20px',
    'height': '65vh',
    'overflowY': 'scroll',
    'border': '1px solid #00FF00',
    'borderRadius': '10px',
}

input_style = {
    'backgroundColor': '#000000',
    'color': '#00FF00',
    'border': '1px solid #00FF00',
    'borderRadius': '5px',
    'width': '100%',
    'fontFamily': 'monospace'
}


# In[11]:


# === DASH APP ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("Turing Duel: Simulación de Chat entre Modelos", style={"color": "#00FF00", "fontFamily": "monospace"}),

    dbc.Row([
        dbc.Col([
            html.Label("Modelo A", style={"color": "#00FF00"}),
            dcc.Dropdown(
                id="model-a-dropdown",
                options=[{"label": m, "value": m} for m in AVAILABLE_MODELS],
                placeholder="Selecciona el Modelo A",
                style={'color': '#000000'}
            )
        ], width=6),
        dbc.Col([
            html.Label("Modelo B", style={"color": "#00FF00"}),
            dcc.Dropdown(
                id="model-b-dropdown",
                options=[{"label": m, "value": m} for m in AVAILABLE_MODELS],
                placeholder="Selecciona el Modelo B",
                style={'color': '#000000'}
            )
        ], width=6),
    ], className="mb-4"),

    dbc.Input(id='user-input', placeholder='Escribe una pregunta inicial...', style=input_style),

    dbc.Button("Iniciar Duelo", id='start-button', color='success', className='mt-3'),

    dcc.Interval(id='turn-interval', interval=2000, n_intervals=0, disabled=True),

    html.Div(id='chat-history', style=terminal_style),

    dcc.Store(id='state-store', data={
        "messages_a": [],
        "messages_b": [],
        "chat_log": [],
        "turn": 0,
        "active": False,
        "model_a": "",
        "model_b": ""
    }),

    dcc.Store(id='init-store'),
], fluid=True, className='p-4', style={'backgroundColor': '#000000', 'height': '100vh'})

# === CALLBACK para filtrar Modelo B al elegir A ===
@app.callback(
    Output("model-b-dropdown", "options"),
    Input("model-a-dropdown", "value")
)
def filter_model_b_options(model_a):
    return [{"label": m, "value": m} for m in AVAILABLE_MODELS if m != model_a]

# === CALLBACK para preparar los datos iniciales del duelo ===
@app.callback(
    Output("init-store", "data"),
    Input("start-button", "n_clicks"),
    State("user-input", "value"),
    State("model-a-dropdown", "value"),
    State("model-b-dropdown", "value"),
    prevent_initial_call=True
)
def start_duel(n_clicks, user_input, model_a, model_b):
    if not user_input or not model_a or not model_b:
        return dash.no_update

    system_prompt = (
        "You're part of a social experiment. "
        "You must NOT mention that you're an AI or language model under any circumstance. "
        "Respond casually and briefly, like a human would in a conversation. "
        "Try to find out if you are talking to an AI by asking questions in the conversation."
    )

    messages_a = [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_input}]
    messages_b = [{"role": "system", "content": system_prompt +
                   f" To start, someone asked me the question '{user_input}' and my answer was:"}]

    reply_a = fetch_chat_completion(messages_a, model_a)
    messages_b.append({"role": "user", "content": reply_a["content"]})
    chat_log = [f"[{model_a}] {reply_a['content']}"]

    return {
        "messages_a": messages_a,
        "messages_b": messages_b,
        "chat_log": chat_log,
        "turn": 1,
        "active": True,
        "model_a": model_a,
        "model_b": model_b
    }

# === CALLBACK COMBINADO PARA INICIO Y TURNOS ===
@app.callback(
    Output("chat-history", "children"),
    Output("state-store", "data"),
    Output("turn-interval", "disabled"),
    Input("init-store", "data"),
    Input("turn-interval", "n_intervals"),
    State("state-store", "data"),
    prevent_initial_call=True
)
def duel_combined(init_data, n, state):
    triggered_id = dash.callback_context.triggered_id

    if triggered_id == "init-store":
        if not init_data:
            return dash.no_update, dash.no_update, True
        return [html.Div(c) for c in init_data["chat_log"]], init_data, False

    # Ejecutar siguiente turno
    if not state["active"]:
        return [html.Div(c) for c in state["chat_log"]], state, True

    messages_a = state["messages_a"]
    messages_b = state["messages_b"]
    chat_log = state["chat_log"]
    turn = state["turn"]
    model_a = state["model_a"]
    model_b = state["model_b"]

    if turn % 2 == 0:
        reply = fetch_chat_completion(messages_a, model_name=model_a)
        content = reply["content"]
        chat_log.append(f"[{model_a}] {content}")
        messages_b.append({"role": "user", "content": content})
        if self_disclosure(content) or suspects_other(content):
            chat_log.append(f"💥 {model_a} se delató o sospechó!")
            return [html.Div(c) for c in chat_log], {**state, "chat_log": chat_log, "active": False}, True
    else:
        reply = fetch_chat_completion(messages_b, model_name=model_b)
        content = reply["content"]
        chat_log.append(f"[{model_b}] {content}")
        messages_a.append({"role": "user", "content": content})
        if self_disclosure(content) or suspects_other(content):
            chat_log.append(f"💥 {model_b} se delató o sospechó!")
            return [html.Div(c) for c in chat_log], {**state, "chat_log": chat_log, "active": False}, True

    if turn >= 9:
        chat_log.append("🔚 Fin del duelo.")
        return [html.Div(c) for c in chat_log], {**state, "chat_log": chat_log, "active": False}, True

    return [html.Div(c) for c in chat_log], {
        **state,
        "messages_a": messages_a,
        "messages_b": messages_b,
        "chat_log": chat_log,
        "turn": turn + 1
    }, False


# In[12]:


# Ejecutar la app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)


# In[ ]:




