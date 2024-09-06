import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os 
from dash import callback_context


# Leer el archivo CSV
df = pd.read_csv('/home/azureuser/proyects/data_lake/dash/dash-quality_data/data/data_num_registros.csv')

# Extraer los datos
sistemas = df['SISTEMA'].tolist()
num_tablas = df['NUMERO TOTAL DE TABLAS'].tolist()
num_tablas_vacias = df['NUMERO DE TABLAS VACIAS'].tolist()
porcentaje_tablas_vacias = df['PORCENTAJE DE TABLAS VACIAS'].tolist()
#SLATE
# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

app.layout = html.Div([
    #BANNER
    html.Div(className="banner"),
    #TABLAS VACIAS POR SISTEMAS
    html.Div([
        #Columna 1:
        html.Div( style={
                 'width': 510,
                 'margin-left': 0,
                 'margin-top': 5,
                 'margin-bottom': 5,
                 'background-color': 'white',
                 'border-radius': 10,
                 'height':'auto'
             }),
        #Columna 2: 
        html.Div([
                # Contenedor principal
            html.Div([
                    # Parte superior 
            html.Div([
                html.Div([
                    dbc.Button("ALL", color="primary", size="sm", id='select-all-button', n_clicks=0),
                    dbc.Button("DELETE", color="secondary", size="sm", id='deselect-all-button', n_clicks=0),
                ], style={'display': 'flex', 'justify-content': 'flex-end', 'margin-top': 5}),
                html.B("TABLAS VACÍAS POR SISTEMA", style={'fontSize': '20px'}),
                html.Hr()
                ], style={'height': 90, 'width': 1020, 'textAlign': 'center' }), 
                    
                    # Parte inferior 
            html.Div([
                dcc.Graph(id='bar-chart')
                ], style={'height': "auto", 'width': 1020 }),
                ], style={'display': 'flex', 'flex-direction': 'column'}),
            
                # Parte lateral 
                html.Div([
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    dbc.Checklist(
                        id='sistemas-checklist',
                        options=[{'label': sistema, 'value': sistema} for sistema in sistemas],
                        value=sistemas[:],
                        inline=False,
                        label_checked_style={"color": "black"},
                        input_checked_style={
                            "backgroundColor": "#2F82B1",
                            "borderColor": "#2F82B1",})
                    ], style={'height': "auto", 'width': 80})
            ], style={
                'display': 'flex',
                'width': 1100,
                'margin-left': 10,
                'margin-right': 10,
                'margin-top': 5,
                'margin-bottom': 5,
                'background-color': 'white',
                'border-radius': 10,
                'height':'auto'
            })
        ],
        style={'display': 'flex'},
        className="contains_graph1"),
    # VALORES NULOS
    html.Div(
        className="contains_graph2"
    )
    ]
)

#GRAFICO TABLAS VACIAS POR SISTEMA
@app.callback(
    Output('sistemas-checklist', 'value'),
    [Input('select-all-button', 'n_clicks'),
     Input('deselect-all-button', 'n_clicks')]
)
def update_checklist(select_all_clicks, deselect_all_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        return sistemas[:]
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'select-all-button':
            return sistemas[:]
        elif button_id == 'deselect-all-button':
            return []

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('sistemas-checklist', 'value')]
)
def update_graph(value):
    # Filtrar los datos según los sistemas seleccionados
    filtered_df = df[df['SISTEMA'].isin(value)]
    sistemas = filtered_df['SISTEMA'].tolist()
    num_tablas = filtered_df['NUMERO TOTAL DE TABLAS'].tolist()
    num_tablas_vacias = filtered_df['NUMERO DE TABLAS VACIAS'].tolist()
    porcentaje_tablas_vacias = filtered_df['PORCENTAJE DE TABLAS VACIAS'].tolist()
    fig = go.Figure()

    # Gráfica de número de tablas
    fig.add_trace(go.Bar(
        y=sistemas,
        x=num_tablas,
        orientation='h',
        name='Tablas totales',
        marker_color='#3E7CB1'
    ))

    # Gráfica de número de tablas vacías
    fig.add_trace(go.Bar(
        y=sistemas,
        x=num_tablas_vacias,
        orientation='h',
        name='Tablas vacías',
        marker_color='#81A4CD'
    ))

    # Añadir etiquetas y título
    fig.update_layout(
        margin=dict(t=30, b=0, l=10, r=0),
        #title='Tablas vacías por sistema'.upper(),
        yaxis_title='Sistemas',
        xaxis_title='Tablas vacias',
        barmode='overlay',
        #legend_title_text='Seleccione:',
        height=745,
        width=1020,
        #paper_bgcolor="#F3F2F2"
        )

    # Añadir los valores en las barras
    for i in range(len(sistemas)):
        fig.add_annotation(
            x=num_tablas[i],
            y=sistemas[i],
            text=f'{num_tablas_vacias[i]}|{num_tablas[i]}<br>{porcentaje_tablas_vacias[i]}%',
            showarrow=False,
            font=dict(size=13),
            xanchor='left'
        )

    return fig

#GRAFICO VALORES NULOS 

if __name__ == '__main__':
    app.run_server(debug=True)