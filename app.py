#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Turing.ipynb

import dash
from dash import dcc, html, Output, Input, State#, ctx
import dash_bootstrap_components as dbc
from styles import terminal_style, input_style
from logic import fetch_chat_completion, self_disclosure, suspects_other
from config import AVAILABLE_MODELS
from callbacks import register_callbacks


# In[2]:


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

register_callbacks(app)


# In[3]:


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

