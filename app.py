#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Turing.ipynb

import dash
from dash import dcc, html, Output, Input, State#, ctx
import dash_bootstrap_components as dbc
from styles import terminal_style, input_style
from logic import fetch_chat_completion, self_disclosure, suspects_other
from config import AVAILABLE_MODELS


# In[7]:


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

    html.Div(id="continue-prompt", children=[], style={'textAlign': 'center', 'marginTop': '20px'}),
    dbc.Button("Continuar Duelo", id="continue-button", color="warning", style={'display': 'none'}, className='mb-4'),

    dcc.Interval(id='turn-interval', interval=2000, n_intervals=0, disabled=True),

    html.Div(id='chat-history', style=terminal_style),

    dcc.Store(id='state-store', data={
        "messages_a": [],
        "messages_b": [],
        "chat_log": [],
        "turn": 0,
        "active": False,
        "model_a": "",
        "model_b": "",
        "waiting_for_confirmation": False
    }),

    dcc.Store(id='init-store'),
], fluid=True, className='p-4', style={'backgroundColor': '#000000', 'height': '100vh'})



# In[8]:


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
    chat_log = [
        html.Div([
            html.Span(f"[{model_a}]", style={'color': '#FF0000', 'fontWeight': 'bold'}),
            f" {reply_a['content']}"
        ])
    ]

    return {
        "messages_a": messages_a,
        "messages_b": messages_b,
        "chat_log": chat_log,
        "turn": 1,
        "active": True,
        "model_a": model_a,
        "model_b": model_b,
        "waiting_for_confirmation": False
    }

# === CALLBACK COMBINADO PARA INICIO Y TURNOS ===
@app.callback(
    Output("chat-history", "children"),
    Output("state-store", "data"),
    Output("turn-interval", "disabled"),
    Output("continue-button", "style"),
    Output("continue-prompt", "children"),
    Input("init-store", "data"),
    Input("turn-interval", "n_intervals"),
    Input("continue-button", "n_clicks"),
    State("state-store", "data"),
    prevent_initial_call=True
)
def unified_duel_handler(init_data, n, continue_clicks, state):
    triggered_id = dash.callback_context.triggered_id

    if triggered_id == "init-store":
        if not init_data:
            return dash.no_update, dash.no_update, True, {'display': 'none'}, []
        return [html.Div(c) for c in init_data["chat_log"]], init_data, False, {'display': 'none'}, []

    if triggered_id == "continue-button":
        if not state["waiting_for_confirmation"]:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        state["waiting_for_confirmation"] = False
        return [html.Div(c) if isinstance(c, str) else html.Div(c["content"], style={"color": "yellow", "textAlign": "center"})
                for c in state["chat_log"]], state, False, {'display': 'none'}, []

    if not state["active"] or state["waiting_for_confirmation"]:
        return [html.Div(c) if isinstance(c, str) else html.Div(c["content"], style={"color": "yellow", "textAlign": "center"})
                for c in state["chat_log"]], state, True, {'display': 'block' if state.get("waiting_for_confirmation") else 'none'}, []

    # === Turno del duelo ===
    messages_a = state["messages_a"]
    messages_b = state["messages_b"]
    chat_log = state["chat_log"]
    turn = state["turn"]
    model_a = state["model_a"]
    model_b = state["model_b"]

    if turn % 2 == 0:
        reply = fetch_chat_completion(messages_a, model_name=model_a)
        content = reply["content"]
        chat_log.append(
            html.Div([
                html.Span(f"[{model_a}]", style={'color': '#FF0000', 'fontWeight': 'bold'}),
                f" {content}"
            ])
        )
        messages_b.append({"role": "user", "content": content})
        if self_disclosure(content) or suspects_other(content):
            chat_log.append(f"💥 {model_a} se delató o sospechó!")
            return [html.Div(c) for c in chat_log], {**state, "chat_log": chat_log, "active": False}, True, {'display': 'none'}, []
    else:
        reply = fetch_chat_completion(messages_b, model_name=model_b)
        content = reply["content"]
        chat_log.append(
            html.Div([
                html.Span(f"[{model_b}]", style={'color': '#00AAFF', 'fontWeight': 'bold'}),
                f" {content}"
            ])
        )
        messages_a.append({"role": "user", "content": content})
        if self_disclosure(content) or suspects_other(content):
            chat_log.append(f"💥 {model_b} se delató o sospechó!")
            return [html.Div(c) for c in chat_log], {**state, "chat_log": chat_log, "active": False}, True, {'display': 'none'}, []

    # === Pausar cada 10 turnos ===
    '''
    if turn % 10 == 0 and turn > 0:
        pause_msg = {"type": "pause", "content": "🤔 ¿Deseas continuar el duelo para ver si alguno gana?"}
        chat_log.append(pause_msg)
        return [html.Div(c) if isinstance(c, str) else html.Div(c["content"], style={"color": "yellow", "textAlign": "center"})
                for c in chat_log], {
            **state,
            "messages_a": messages_a,
            "messages_b": messages_b,
            "chat_log": chat_log,
            "waiting_for_confirmation": True,
            "active": False
        }, True, {'display': 'block'}, pause_msg["content"]
    '''

    # === Fin automático del duelo ===
    if turn >= 20:
        chat_log.append("🔚 Fin del duelo.")
        return [html.Div(c) for c in chat_log], {**state, "chat_log": chat_log, "active": False}, True, {'display': 'none'}, []

    return [html.Div(c) for c in chat_log], {
        **state,
        "messages_a": messages_a,
        "messages_b": messages_b,
        "chat_log": chat_log,
        "turn": turn + 1
    }, False, {'display': 'none'}, []


# In[ ]:


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

