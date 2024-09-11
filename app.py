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
import math
import random

# Leer el archivo CSV
df = pd.read_csv('dash-quality_data/data/data_num_registros.csv')
path_all_tables = '/home/ale1726/proyects/dash/dash-quality_data/data/all_tables'
descripcion = df['NUMERO DE TABLAS VACIAS'].describe()

# Extraer los datos bar char
sistemas_list= df['SISTEMA'].tolist()
num_tablas = df['NUMERO TOTAL DE TABLAS'].tolist()
num_tablas_vacias = df['NUMERO DE TABLAS VACIAS'].tolist()
porcentaje_tablas_vacias = df['PORCENTAJE DE TABLAS VACIAS'].tolist()

#Extraer datos par dispersion char

df_all_tables = [os.path.join(path_all_tables,archivo) for archivo in os.listdir(path_all_tables)]
list_df_1 = [pd.read_csv(df2) for df2 in df_all_tables]

descripcion_dict = descripcion.to_dict()

encabezado = [
    html.Thead(html.Tr([html.Th("Estadística"), html.Th("Valor")]))
]
estadisticas=['Numero de sistemas',
              'Promedio de tablas vacias',
              'Desviacion estandar',
              'Numero minimo de tablas vacias',
              "El '25%' del total de tablas",
              "El '50%' del total de tablas",
              "El '75%' del total de tablas",
              'Numero maximo de tablas vacias'
              ]
rows = []
for key, value in zip(estadisticas,descripcion_dict.values()):
    rows.append(html.Tr([html.Td(key), html.Td(round(value,2))]))

cuerpo_tabla = [html.Tbody(rows)]

table_graph1 = dbc.Table(encabezado + cuerpo_tabla, color='blue', bordered=True, dark=True, hover=True, responsive=True, striped=True)
# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

app.layout = html.Div([
    #BANNER
    html.Div("ANÁLISIS DE CALIDAD DE DATOS DE LOS SISTEMAS DE CLIENTES", className="banner"),
    html.Div("TABLAS VACIAS POR SISTEMA", className="banner_2"),
    #TABLAS VACIAS POR SISTEMAS
    html.Div([
        #Columna 1:
        html.Div([
            html.Div([
                dbc.Card(
                        dbc.CardBody(
                                [
                                    html.P('NÚMERO TOTAL DE TABLAS: ', style={'font-size': '17px', 'color':'white', 'font-weight': 'bold', 'textAlign': 'justify'}),
                                    html.P(f'{int(sum(num_tablas))}', style={'font-size': '50px', 'color':'white', 'font-weight': 'bold','textAlign': 'center'})
                                ]
                            ),
                            style={
                                'display': 'flex',
                                'width': "45%",
                                'height': '80%',
                                'background-color': '#0081a7',
                                'margin': '5px',
                                'border-radius': '15%'
                            }
                        ),
                dbc.Card(
                        dbc.CardBody(
                                [
                                    html.P('NÚMERO TOTAL DE TABLAS VACÍAS: ', style={'font-size': '17px', 'color':'white', 'font-weight': 'bold', 'textAlign': 'left'}),
                                    html.P(f'{int(sum(num_tablas_vacias))}', style={'font-size': '50px', 'color':'white','font-weight': 'bold', 'textAlign': 'center'})
                                ]
                            ),
                            style={
                                'display': 'flex',
                                'width': "45%",
                                'height': '80%',
                                'background-color': '#00afb9',
                                'margin': '5px',
                                'border-radius': '15%'
                            }
                        )
                ],
                    style={
                        'margin-top': 30,
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        #'margin': '10px',
                        'height': '30%'
                    }   
                     ),
            html.Div([
                table_graph1
            ],
                style={
                'margin-left': 15,
                'margin-right': 15,
                'margin-top': 10,
                'margin-bottom': 5,
                }),
        ],style={
                'width': "25%",
                'margin-left': 0,
                'margin-top': 0,
                'margin-bottom': 5,
                'background-color': 'white',
                'border-radius': 10,
                'height': 'auto'
            }),
        #Columna 2: 
        html.Div([
                # Contenedor principal
            html.Div([
                    # Parte superior 
            html.Div([
                html.Div([
                    html.B("FRECUENCIA DE TABLAS VACÍAS POR SISTEMA", style={'fontSize': '22px', 'fontFamily': 'sans-serif', 'textTransform': 'uppercase', 'color':'#A0A0A0'})
                    ],
                    style={'height': '90%', 'width': '65%',  'textAlign': 'right', 'margin-top': 5}),
                
                html.Div([
                    dbc.Button("ALL", color="primary", size="sm", id='select-all-button', n_clicks=0),
                    dbc.Button("DELETE", color="secondary", size="sm", id='deselect-all-button', n_clicks=0),
                    ],
                    style={'height': '90%', 'width': '35%', 'textAlign': 'right', 'margin-top': 5}),
            ], style={'display': 'flex', 'height': '5%', 'width': '100%', 'flexDirection': 'row'}),
            html.Div(html.Hr(),style={'height': '3%', 'width': '100%'}),                   
                # Parte inferior 
            html.Div([
                dcc.Graph(id='bar-chart', config={'responsive': True})
                ], style={'height': "93%"}),
                ], style={'height':'100%','display': 'flex', 'flex-direction': 'column', 'width': '93%', 'background-color':'white'}),
            
                # Parte lateral 
                html.Div([
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    dbc.Checklist(
                        id='sistemas-checklist',
                        options=[{'label': sistema, 'value': sistema} for sistema in sistemas_list],
                        value=sistemas_list[:],
                        inline=False,
                        label_checked_style={"color": "black"},
                        input_checked_style={
                            "backgroundColor": "#2F82B1",
                            "borderColor": "#2F82B1",})
                    ], style={'height': "100%", 'width': '7%'})
            ], style={
                'display': 'flex',
                'width': '75%',
                'margin-left': 10,
                'margin-right': 10,
                'margin-top': 5,
                'margin-bottom': 5,
                'background-color': 'white',
                'border-radius': 10,
                'height':'100%'
            })
        ],
        style={'display': 'flex'},
        className="contains_graph1"),
    # VALORES NULOS
    html.Div([
        html.Div([
            "VALORES NULOS POR SISTEMA"
            ],className="banner_2"),
        
        html.Div([
                dbc.Tabs(
                    id='bar-chart-sistema-tabs',
                    active_tab=f'{sistemas_list[0]}',
                    class_name='d-flex justify-content-center w-100',
                    children=[
                        dbc.Tab(label=str(sistema), tab_id=f'{sistema}') for sistema in sistemas_list
                    ]
                ), 
            ], 
            style={'height': '10%',
                   'display': 'flex', 
                   'flex-direction': 'column', 
                   'width': '100%', 
                   'background-color': 'white'}),
        
        html.Div(dcc.Graph(id='dispersion-chart', config={'responsive': True}),
                style={'height':'80%','display': 'flex', 'flex-direction': 'column', 'width': '60%' }),
        html.Div(
            [
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Div(id='table-container', style={'margin-left': 100, 'margin-right': 100})
            ],
            style={'height': '80%', 'display': 'flex', 'flex-direction': 'column', 'width': '40%', 'background-color': 'white'}
            )
        ],
        style={'display': 'flex', 'flex-wrap': 'wrap'},
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
        return sistemas_list[:]
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'select-all-button':
            return sistemas_list[:]
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
        name='Tablas no vacias',
        marker_color='#3E7CB1',
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
        margin=dict(t=20, b=0, l=10, r=0),
        #title='Tablas vacías por sistema'.upper(),
        yaxis_title='Sistemas',
        xaxis_title='Tablas vacias',
        barmode='overlay',
        height=700,
        #width=1200,
        #paper_bgcolor="#F3F2F2"
        )

    # Añadir los valores en las barras
    for i in range(len(sistemas)):
        fig.add_annotation(
            x=num_tablas[i],
            y=sistemas[i],
            text=f'{num_tablas_vacias[i]}|{num_tablas[i]}<br>{porcentaje_tablas_vacias[i]}%',
            showarrow=False,
            font=dict(size=14,
                    family="sans-serif",  # Tipo de fuente
                    color="#757575"  # Color del texto
                    ),
            xanchor='left'
        )
    return fig
# Construccion dataframe para dispersion chart
list_df_table = []
for i, df1 in enumerate(list_df_1):
    tables = df1['TABLE_NAME'].unique()
    dict_temp = {}
    df['NUM_NULLS']=df1['NUM_NULLS'].fillna(0)
    for table in tables:
        conteo = df1['NUM_NULLS'][df1['TABLE_NAME'] == table]
        descripcion = df1['NUM_NULLS'][df1['TABLE_NAME'] == table].describe()
        filter_1 = df1[df1['TABLE_NAME'] == table]
        min_value = filter_1['NUM_NULLS'].min()
        max_value = filter_1['NUM_NULLS'].max()
        min_value_2 = "{:,.2f}".format(descripcion['min'])
        max_value_2 = "{:,.2f}".format(descripcion['max'])
        # Asegurarse de que haya valores mínimos y máximos
        columns_min = filter_1['COLUMN_NAME'][filter_1['NUM_NULLS'] == min_value].iloc[0] if not filter_1['COLUMN_NAME'][filter_1['NUM_NULLS'] == min_value].empty else None
        columns_max = filter_1['COLUMN_NAME'][filter_1['NUM_NULLS'] == max_value].iloc[0] if not filter_1['COLUMN_NAME'][filter_1['NUM_NULLS'] == max_value].empty else None
        
        dict_temp[table] = {
            'VALORES NULOS': sum(conteo),
            'MEDIA': sum(conteo) / len(conteo) if len(conteo) > 0 else 0,
            'COUNT': "{:,.2f}".format(descripcion['count']),
            'STD': "{:,.2f}".format(descripcion['std']),
            'COL_MIN': columns_min,
            'COL_MAX': columns_max,
            'MIN': f"{min_value_2}" if columns_min is not None else 'N/A',
            'MAX': f"{max_value_2}" if columns_max is not None else 'N/A'
        }
    #df_result
    df_result = pd.DataFrame.from_dict(dict_temp, orient='index')
    df_result = df_result.reset_index()
    df_result.columns = ['TABLE_NAME', 'VALORES_NULOS', 'MEDIA', 'COUNT', 'STD','COL_MIN','COL_MAX','MIN', 'MAX']
    df_result['SISTEMA'] = ' '.join(df1['SISTEMA'].unique())
    list_df_table.append(df_result)

@app.callback(
    Output('dispersion-chart', 'figure'),
    [Input('bar-chart-sistema-tabs', 'active_tab')]
)
#GRAFICO VALORES NULOS 
def update_graph_vall_null(select_system):
    colors = ["#0a2239","#336c81","#1d84b5","#00eaff","#00ffcc",'#00ff48', '#f2ff00', '#ff8009', '#f07167']
    # Concatenar los DataFrames
    df_3 = pd.concat(list_df_table, ignore_index=True)
    names =  df_3['SISTEMA'].unique().tolist()
    filtred = select_system # Extraer el nombre del sistema del tab_id
    df_3 = df_3[df_3['SISTEMA'] == filtred]
    #df_3 = df_3.sort_values(by='TABLE_NAME')
    df_3= df_3.sample(frac=1).reset_index(drop=True)
    # Formatear las columnas 'VALORES_NULOS' y 'MEDIA' como cadenas con dos decimales
    df_3['VALORES_NULOS_STR'] = df_3['VALORES_NULOS'].apply(lambda x: "{:,.2f}".format(x))
    df_3['MEDIA_STR'] = df_3['MEDIA'].apply(lambda x: "{:,.2f}".format(x))
    # Crear la columna 'text' con la información formateada
    df_3['text'] = df_3.apply(lambda row: (f"Sistema: {row['SISTEMA']}<br>"
                                           f"Tabla:  {row['TABLE_NAME']}<br>"
                                           f"Valores nulos: {row['VALORES_NULOS_STR']}<br>"
                                           f"Promedio de valores nulos: {row['MEDIA_STR']}<br>"
                                           f"Desviacion estandar: {row['STD']}<br>"
                                           f"Numero de columnas: {row['COUNT']}<br>"
                                           f"Numero minimo de val. nuls: {row['MIN']}<br>"
                                           f"Numero maximo de val. nuls: {row['MAX']}<br>"
                                           f"Col min {row['COL_MIN']}<br>"
                                           f"Col max {row['COL_MAX']}")
                                           , axis=1)
    # Calcular el tamaño de la burbuja, manejando NaN y valores negativos
    df_3['bubble_size'] = df_3['VALORES_NULOS'].apply(lambda x: math.sqrt(x) if pd.notna(x) and x >= 0 else 0)
    df_3['bubble_size'] = df_3['bubble_size'].apply(lambda x: 150 if x== 0 else x)
    # Determinar el factor de referencia de tamaño para las burbujas1
    sizeref = 2. * max(df_3['bubble_size']) / (50**2)
    # Crear un diccionario que contiene DataFrames separados por sistema
    sistemas_name = df_3['SISTEMA'].unique()
    data = {sistema: df_3[df_3['SISTEMA'] == sistema] for sistema in sistemas_name}
    color_map = {sistema: colors[random.randint(0,len(colors)-1)] for _, sistema in enumerate(sistemas_name)}
    # Crear el gráfico usando Plotly
    fig = go.Figure()
    for sistema, datos in data.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(df_3))),  # Ajustar el eje X a la longitud de los datos
            y=datos['VALORES_NULOS'],
            name=sistema,
            text=datos['text'],
            marker=dict(size=datos['bubble_size'],
                        color=color_map[sistema])
        ))
    fig.update_traces(mode='markers', marker=dict(sizemode='area', sizeref=sizeref, line_width=2))
    # Ajustar el diseño
    fig.update_layout(
        #autosize=True,
        height=650,        #alto
        #width=600,        #ancho total
        showlegend=True,       
        title={
            'text': f'DISPERSION DE VALORES NULOS POR TABLA EN EL SISTEMA: {filtred}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 22,
                'family': 'sans-serif',
                 'color': '#A0A0A0'}
        },
        yaxis_title='Valores nulos',
        xaxis_title='Tablas'
    )
    return fig


@app.callback(
    Output('table-container', 'children'),
    [Input('bar-chart-sistema-tabs', 'active_tab')]
)
def update_table(active_tab):
    df_4 = pd.concat(list_df_table, ignore_index=True)
    filtred = active_tab # Extraer el nombre del sistema del tab_id
    df_4 = df_4[df_4['SISTEMA'] == filtred]

    val_max = df_4['VALORES_NULOS'].max()
    val_min = df_4['VALORES_NULOS'].min()
    media = df_4['VALORES_NULOS'].mean()

    tabla_descripcion = {
        'Descripción': [
            f'Numero maximo de valores nulos en {filtred}',
            f'Numero minimo de valores nulos en {filtred}',
            f'Tabla con mayor numero de valores nulos en {filtred}',
            f'Tabla con menor numero de valores nulos en {filtred}',
            f'Promedio de valores nulos en {filtred}',
            f'Numero de tablas en {filtred}'
        ],
        'Valor': [
            "{:,.1f}".format(val_max),
            "{:,.1f}".format(val_min),
            df_4['TABLE_NAME'][df_4['VALORES_NULOS'] == val_max].values[0],
            df_4['TABLE_NAME'][df_4['VALORES_NULOS'] == val_min].values[0],
            "{:,.1f}".format(media),
            len(df_4['TABLE_NAME'].unique())
        ]
    }

    table_header = [
        html.Thead(html.Tr([html.Th("Descripción"), html.Th("Valor")]))
    ]

    table_body = [
        html.Tbody([
            html.Tr([html.Td(tabla_descripcion['Descripción'][i]), html.Td(tabla_descripcion['Valor'][i])])
            for i in range(len(tabla_descripcion['Descripción']))
        ])
    ]

    table = dbc.Table(table_header + table_body, bordered=True, dark=True, hover=True, responsive=True, striped=True)

    return table

if __name__ == '__main__':
    app.run_server(debug=True, port = 8051)