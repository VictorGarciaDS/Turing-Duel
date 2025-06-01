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

# === CONFIGURACIÓN Y DATOS ===
load_dotenv()

OPEN_API_ENDPOINT = "https://models.inference.ai.azure.com"
OPEN_API_KEY = os.getenv("GITHUB_TOKEN")

AVAILABLE_MODELS = pd.read_csv("conversational_models_sorted.csv")["model"].dropna().unique().tolist()

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
    except openai.BadRequestError as e:
#        print(f"Error al llamar al modelo {model_name}: {e}")
        return {"role": "assistant", "content": f"[ERROR: Modelo inválido: {model_name}]"}

def self_disclosure(text):
    return any(k in text.lower() for k in DETECTION_KEYWORDS)

def suspects_other(text):
    return any(k in text.lower() for k in SUSPECT_KEYWORDS)

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
                placeholder="Selecciona el Modelo B",
                style={'color': '#000000'}
            )
        ], width=6)
    ], className="mb-4"),

    html.Div(id='chat-history', style=terminal_style),

    dbc.InputGroup([
        dbc.Input(id='user-input', placeholder='Escribe una pregunta inicial...', style=input_style),
        dbc.Button("Enviar", id='send-button', color='success')
    ], className='mt-3'),

    dcc.Store(id='stored-messages', data=[]),
], fluid=True, className='p-4', style={'backgroundColor': '#000000', 'height': '100vh'})

# === CALLBACKS ===

@app.callback(
    Output("model-b-dropdown", "options"),
    Input("model-a-dropdown", "value")
)
def update_model_b_options(model_a):
    return [{"label": m, "value": m} for m in AVAILABLE_MODELS if m != model_a]

@app.callback(
    Output('chat-history', 'children'),
    Output('stored-messages', 'data'),
    Input('send-button', 'n_clicks'),
    State('user-input', 'value'),
    State('model-a-dropdown', 'value'),
    State('model-b-dropdown', 'value'),
    prevent_initial_call=True
)
def simulate_duel(n_clicks, user_input, model_a, model_b):
    if not user_input or not model_a or not model_b:
        return [html.Div("⚠️ Por favor completa los campos.")], []
    
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

    reply_b = fetch_chat_completion(messages_b, model_b)
    messages_a.append({"role": "user", "content": reply_b["content"]})

    chat_log = []

    # Conversación por turnos
    for turn in range(10):  # para mantenerlo corto
        if turn % 2 == 0:
            reply = fetch_chat_completion(messages_a, model_name=model_a)
            content = reply["content"]
            chat_log.append(html.Div(f"[{model_a}] {content}"))

            if self_disclosure(content) or suspects_other(content):
                chat_log.append(html.Div(f"💥 {model_a} se delató o sospechó!"))
                break

            messages_b.append({"role": "user", "content": content})

        else:
            reply = fetch_chat_completion(messages_b, model_name=model_b)
            content = reply["content"]
            chat_log.append(html.Div(f"[{model_b}] {content}"))

            if self_disclosure(content) or suspects_other(content):
                chat_log.append(html.Div(f"💥 {model_b} se delató o sospechó!"))
                break

            messages_a.append({"role": "user", "content": content})

#        print("Turno ", turn, ": ", content)
        time.sleep(0.5)  # Para evitar throttling

    # Al final, retornar tanto el chat visual como los textos
    return chat_log, [div.children for div in chat_log]
# Ejecutar la app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)