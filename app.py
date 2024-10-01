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
import plotly.express as px

# Leer el archivo CSV
df = pd.read_csv('/home/ale1726/proyects/dash/dash-quality_data/data/data_num_registros.csv')
names =  pd.read_csv('/home/ale1726/proyects/dash/dash-quality_data/data/names.csv')
df_registros=pd.read_csv('/home/ale1726/proyects/dash/dash-quality_data/data/data_graph_registro.csv')
path_all_tables = '/home/ale1726/proyects/dash/dash-quality_data/data/all_tables'
df_storage=pd.read_csv('/home/ale1726/proyects/dash/dash-quality_data/data/almacenamiento.csv')
df_registros_unicos=pd.read_csv('/home/ale1726/proyects/dash/dash-quality_data/data/conteos.csv')


# Formatear las columnas 'VALORES_NULOS' y 'MEDIA' como cadenas con dos decimales
df_registros_unicos['NUM_DISTINCT_STR'] = df_registros_unicos['NUM_DISTINCT'].apply(lambda x: "{:,.2f}".format(x))

# Crear la columna 'text' con la información formateada
df_registros_unicos['text_num_distinct'] = df_registros_unicos.apply(lambda row: (
                                       f"SISTEMA: {row['SISTEMA']}<br>"
                                       f"Tabla:  {row['TABLE_NAME']}<br>"
                                       f"Columna: {row['COLUMN_NAME']}<br>"
                                       f"Longitud columna: {row['NUM_DISTINCT_STR']}<br>")
                                       , axis=1)

df_registros_unicos['bubble_size_num_distinct'] = df_registros_unicos['NUM_DISTINCT'].apply(lambda x: math.sqrt(x) if pd.notna(x) and x >= 0 else 0)
df_registros_unicos['bubble_size_num_distinct'] = df_registros_unicos['bubble_size_num_distinct'].apply(lambda x: x+20)


# Formatear las columnas 'VALORES_NULOS' y 'MEDIA' como cadenas con dos decimales
df_registros_unicos['AVG_COL_LEN_STR'] = df_registros_unicos['AVG_COL_LEN'].apply(lambda x: "{:,.2f}".format(x))

# Crear la columna 'text' con la información formateada
df_registros_unicos['text_avg_len'] = df_registros_unicos.apply(lambda row: (
                                       f"SISTEMA: {row['SISTEMA']}<br>"
                                       f"Tabla:  {row['TABLE_NAME']}<br>"
                                       f"Columna: {row['COLUMN_NAME']}<br>"
                                       f"Longitud columna: {row['AVG_COL_LEN_STR']}<br>")
                                       , axis=1)

df_registros_unicos['bubble_size_avg_len'] = df_registros_unicos['AVG_COL_LEN'].apply(lambda x: math.sqrt(x) if pd.notna(x) and x >= 0 else 0)
df_registros_unicos['bubble_size_avg_len'] = df_registros_unicos['bubble_size_avg_len'].apply(lambda x: x+20)


# Extraer los datos bar char
sistemas_list= df['SISTEMA'].tolist()
num_tablas = df['NUMERO TOTAL DE TABLAS'].tolist()
num_tablas_vacias = df['NUMERO DE TABLAS VACIAS'].tolist()
porcentaje_tablas_vacias = df['PORCENTAJE DE TABLAS VACIAS'].tolist()

#Extraer datos par dispersion char
df_all_tables = [os.path.join(path_all_tables,archivo) for archivo in os.listdir(path_all_tables)]
list_df_1 = [pd.read_csv(df2) for df2 in df_all_tables]

# TABLA INFORMACION
num_sistema = df['NUMERO DE TABLAS VACIAS'].describe()['count']
media_vacias = df['NUMERO DE TABLAS VACIAS'].describe()['mean']
std_vacias = df['NUMERO DE TABLAS VACIAS'].describe()['std']
q1_vacias= f"{int(df['NUMERO DE TABLAS VACIAS'].describe()['25%'])} tablas vacias"
q2_vacias=f"{int(df['NUMERO DE TABLAS VACIAS'].describe()['50%'])} tablas vacias"
q3_vacias=f"{df['NUMERO DE TABLAS VACIAS'].describe()['75%']} tablas vacias"
min_vacias=df['NUMERO DE TABLAS VACIAS'].describe()['min']
max_vacias=df['NUMERO DE TABLAS VACIAS'].describe()['max']

tabla_vacias_descripcion ={
     'Descripción': ['Numero de sistemas',
              'Promedio de tablas vacias',
              'Desviacion estandar',
              'Numero maximo de tablas vacias',
              'Numero minimo de tablas vacias',
              "El '25%' de los sistemas tienen",
              "El '50%' de los sistemas tienen",
              "El '75%' de los sistemas tienen"              
              ],
        'Valor': [
            num_sistema,
            "{:,.1f}".format(media_vacias),
            "{:,.1f}".format(std_vacias),
            "{:,.1f}".format(max_vacias),
            "{:,.1f}".format(min_vacias),
            q1_vacias,
            q2_vacias,
            q3_vacias
        ]
    }

encabezado = [
    html.Thead(html.Tr([html.Th('Descripción'), html.Th('Valor')]))
]

cuerpo_tabla = [
    html.Tbody([
        html.Tr([html.Td(tabla_vacias_descripcion['Descripción'][i]), html.Td(tabla_vacias_descripcion['Valor'][i])])
        for i in range(len(tabla_vacias_descripcion['Descripción']))
    ])
]


table_graph1 = dbc.Table(encabezado + cuerpo_tabla, bordered=True, dark=True, hover=True, responsive=True, striped=True)
# Crear la aplicación Dash
#Morph MORPH
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

app.layout = html.Div([
    #BANNER
    html.Div([
        html.Img(src='assets/Logo_de_Nafin_white.png', className='logo'),
        html.Div("ANÁLISIS DE CALIDAD DE DATOS DE LOS SISTEMAS DE CLIENTES", className="banner-text")
    ], className="banner"),
    html.Div("TABLAS VACIAS POR SISTEMA", className="banner_2"),
    #TABLAS VACIAS POR SISTEMAS
    html.Div([
        #Columna 1:
        html.Div([
            html.Div([
                dbc.Card(
                        dbc.CardBody(
                                [
                                    html.P('NÚMERO TOTAL DE TABLAS: ', style={'font-size': '15px', 'color':'white', 'font-weight': 'bold', 'textAlign': 'justify'}),
                                    html.P(f'{int(sum(num_tablas))}', style={'font-size': '45px', 'color':'white', 'font-weight': 'bold','textAlign': 'center'})
                                ]
                            ),
                            style={
                                'display': 'flex',
                                'width': "45%",
                                'height': '100%',
                                'background-color': '#0081a7',
                                'margin': '5px',
                                'border-radius': '15%'
                            }
                        ),
                dbc.Card(
                        dbc.CardBody(
                                [
                                    html.P('NÚMERO DE TABLAS VACÍAS: ', style={'font-size': '15px', 'color':'white', 'font-weight': 'bold', 'textAlign': 'justify'}),
                                    html.P(f'{int(sum(num_tablas_vacias))}', style={'font-size': '45px', 'color':'white','font-weight': 'bold', 'textAlign': 'center'})
                                ]
                            ),
                            style={
                                'display': 'flex',
                                'width': "45%",
                                'height': '100%',
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
                        'height': '20%'
                    }   
                     ),
            html.Div([
                table_graph1
            ],
                style={
                'margin-left': 15,
                'margin-right': 15,
                'margin-top': 25,
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
                    html.B("FRECUENCIA DE TABLAS VACÍAS POR SISTEMA", style={'margin-left':5,'margin-top': 5,'fontSize': '22px', 'fontFamily': 'sans-serif', 'textTransform': 'uppercase', 'color':'#A0A0A0'})
                    ],
                    style={'height': '90%', 'width': '65%',  'textAlign': 'right', 'margin-top': 5}),
                
                html.Div([
                    dbc.Button("ALL", color="primary", className="me-1", size="sm", id='select-all-button', n_clicks=0),
                    dbc.Button("DELETE", color="danger", className="me-1", size="sm", id='deselect-all-button', n_clicks=0),
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
                        label_checked_style={"color": "#0046a7"},
                        input_checked_style={
                            "backgroundColor": "#47b3fe",
                            "borderColor": "#4777a6",})
                    ], style={'height': "100%", 'width': '7%',  'margin-left': 5})
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
                        dbc.Tab(label=str(sistema), 
                                tab_id=f'{sistema}',
                                active_label_style={"backgroundColor": "#00D1FF"}) for sistema in sistemas_list
                    ]
                ), 
            ], 
            style={'height': '5%',
                   'display': 'flex', 
                   'flex-direction': 'column', 
                   'width': '100%', 
                   'background-color': 'white'}),
        
        html.Div(dcc.Graph(id='dispersion-chart', style={'width': '95%','height': '90%',
                                                     'margin-top': 15,
                                                     'margin-left': 5, 
                                                     'margin-right': 0}),
                style={'height':'80%','display': 'flex', 'flex-direction': 'column', 'width': '55%' }),
        #columna descripcion
        html.Div(
             [   
            html.Div([
                html.H4(id='name_system', style={'textAlign': 'center','margin-top': 10})
            ], style={'background-color': 'white', 'height': '10%'}),

            html.Div(id='card_info', 
                     style={'justify-content': 'center',  # Centra horizontalmente
                            'align-items': 'center',      # Centra verticalmente 
                            'display': 'flex', 
                            'flex-wrap': 'wrap', 
                            'height': '20%'}),
            html.Br(),
            html.Div(id='table-container', style={'margin-left': 50, 'margin-right': 50,'height': '55%'})
            ],
            style={'height': '80%', 'display': 'flex', 'flex-direction': 'column', 'width': '40%', 'background-color': 'white'}
            )
        ],
        style={'display': 'flex', 'flex-wrap': 'wrap'},
        className="contains_graph2"
    ),
       # NUMERO DE REGISTROS
    html.Div([
        html.Div([
            "NUMERO DE REGISTROS"
            ],className="banner_2"),
        
        html.Div([
                dbc.Tabs(
                    id='bar-chart-sistema-tabs_2',
                    active_tab=f'{sistemas_list[0]}',
                    class_name='d-flex justify-content-center w-100',
                    children=[
                        dbc.Tab(label=str(sistema), 
                                tab_id=f'{sistema}',
                                active_label_style={"backgroundColor": "#00D1FF"}) for sistema in sistemas_list
                    ]
                ), 
            ], 
            style={'height': '5%',
                   'display': 'flex', 
                   'flex-direction': 'column', 
                   'width': '100%', 
                   'background-color': 'white'}),
        
        html.Div(dcc.Graph(id='dispersion-num-registros',style={'width': '95%','height': '90%',
                                                     'margin-top': 15,
                                                     'margin-left': 5, 
                                                     'margin-right': 0} ),
                        style={'height':'80%',
                               'display': 'flex',
                               'flex-direction': 'column',
                               #'border': '1px solid #ddd',
                               #'border-radius': '5%',
                               #'margin-left': 0,  
                               'width': '55%' }),
        
        #columna descripcion
        html.Div(
             [   
            html.Div([
                html.H4(id='name_system_2', style={'textAlign': 'center','margin-top':10})
            ], style={'background-color': 'white', 'height': '10%'}),

            html.Div(id='card_info_registro', 
                     style={'display': 'flex',
                            'justify-content': 'center',  # Centra horizontalmente
                            'align-items': 'center',      # Centra verticalmente 
                            'flex-wrap': 'wrap', 
                            'background-color': 'white', 
                            'height': '20%'}),
            html.Br(),
            html.Div(id='table-container_register',style={'margin-left': 10, 'margin-right': 20, 'height': '55%'})
            ],
            style={'height': '80%', 'display': 'flex', 'flex-direction': 'column', 'width': '40%', 'background-color': 'white'}
            )
        ],
        style={'display': 'flex', 'flex-wrap': 'wrap'},
        className="contains_graph2"
    ),
     # NUMERO DE REGISTROS UNICOS
    html.Div([
        html.Div([
            "NUMERO DE REGISTROS UNICOS"
            ],className="banner_2"),
        
        html.Div([
                dbc.Tabs(
                    id='bar-chart-sistema-tabs_3',
                    active_tab=f'{list(df_registros_unicos["SISTEMA"].unique())[0]}',
                    class_name='d-flex justify-content-center w-100',
                    children=[
                        dbc.Tab(label=str(sistema), 
                                tab_id=f'{sistema}',
                                active_label_style={"backgroundColor": "#00D1FF"}) for sistema in list(df_registros_unicos['SISTEMA'].unique())
                    ]
                ), 
            ], 
            style={'height': '5%',
                   'display': 'flex', 
                   'flex-direction': 'column', 
                   'width': '100%', 
                   'background-color': 'white'}),
        
        html.Div(dcc.Graph(id='dispersion-num-registros_unicos',style={'width': '95%','height': '90%',
                                                     'margin-top': 15,
                                                     'margin-left': 5, 
                                                     'margin-right': 0} ),
                        style={'height':'80%',
                               'display': 'flex',
                               'flex-direction': 'column',
                               #'border': '1px solid #ddd',
                               #'border-radius': '5%',
                               #'margin-left': 0,  
                               'width': '55%' }),
        
        #columna descripcion
        html.Div(
             [   
            html.Div([
                html.H4(id='name_system_3', style={'textAlign': 'center','margin-top':10})
            ], style={'background-color': 'white', 'height': '10%'}),

            html.Div(id='card_info_registro_unicos', 
                     style={'display': 'flex',
                            'justify-content': 'center',  # Centra horizontalmente
                            'align-items': 'center',      # Centra verticalmente 
                            'flex-wrap': 'wrap', 
                            'background-color': 'white', 
                            'height': '20%'}),
            html.Br(),
            html.Div(id='table-container_register_unicos',style={'margin-left': 10, 'margin-right': 20, 'height': '55%'})
            ],
            style={'height': '80%', 'display': 'flex', 'flex-direction': 'column', 'width': '40%', 'background-color': 'white'}
            )
        ],
        style={'display': 'flex', 'flex-wrap': 'wrap'},
        className="contains_graph2"
    ),
    #ALMACENAMIENTO
    html.Div([
        html.Div([
            "Almacenamiento de los sistemas"
        ], className="banner_2"),
        
        html.Div(id='kpis_storage',
                 style={'display': 'flex',
                        'justify-content': 'center',  # Centra horizontalmente
                        'align-items': 'center',      # Centra verticalmente 
                        'background-color': '#ffffff',
                        'height': '20%','width': '100%'}),
        
        html.Div(dcc.Graph(id='treemap_file', style={'width': '95%','height': '95%',
                                                     'margin-top': 15,
                                                     'margin-left': 2, 
                                                     'margin-right': 2}),
                 style={'height': '65%', 
                        'display': 'flex', 
                        'flex-direction': 'column', 
                        'width': '45%',
                        'border-radius': '5%',
                        'margin-left': 50,
                        #box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                        'border': '1px solid #ddd',
                        'background-color': 'white'}),
        
      
        html.Div(dcc.Graph(id='bar_storage', style={'width': '95%','height': '95%',
                                                    'margin-top': 15,
                                                    'margin-left': 2, 
                                                    'margin-right': 2}),
                        style={'height': '65%', 
                        'display': 'flex', 
                        'flex-direction': 'column', 
                        'width': '45%', 
                        'border-radius': '5%',
                        'margin-left': 70,
                        'border': '1px solid #ddd',
                        'background-color': 'white'})
    ],
        style={'display': 'flex', 'flex-wrap': 'wrap'},
        className="contains_graph3"
    ),
    #TIPO DE DATO
    html.Div([
        html.Div([
            "TIPOS DE DATOS EN LOS SISTEMAS DE CLIENTES"
        ], className="banner_2"),
        
        html.Div(id='kpis_type_data',
                 style={'display': 'flex',
                        'justify-content': 'center',  # Centra horizontalmente
                        'align-items': 'center',      # Centra verticalmente 
                        'background-color': '#ffffff',
                        'height': '20%','width': '100%'}),
        
        html.Div(dcc.Graph(id='bar_type_data', style={'width': '95%','height': '95%',
                                                     'margin-top': 15,
                                                     'margin-left': 10, 
                                                     'margin-right': 2}),
                 style={'height': '68%', 
                        'display': 'flex',
                        'width': '80%',
                        #'border-radius': '15%',
                        'margin-left': 130,
                        #box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                        'border': '1px solid #ddd',
                        'background-color': 'white'})
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
    df['NUMERO DE TABLAS NO VACIAS']=df['NUMERO TOTAL DE TABLAS']-df['NUMERO DE TABLAS VACIAS']
    filtered_df = df[df['SISTEMA'].isin(value)]
    sistemas = filtered_df['SISTEMA'].tolist()
    num_tablas = filtered_df['NUMERO TOTAL DE TABLAS'].tolist()
    num_tablas_vacias = filtered_df['NUMERO DE TABLAS VACIAS'].tolist()
    #num_tablas_no_vacias = filtered_df['NUMERO DE TABLAS NO VACIAS'].tolist()
    porcentaje_tablas_vacias = filtered_df['PORCENTAJE DE TABLAS VACIAS'].tolist()
    fig = go.Figure(data=[
    go.Bar(name='Tablas totales', y=sistemas,x=num_tablas,orientation='h',marker_color='#0081a7'),
    go.Bar(name='Tablas Vacías', y=sistemas, x=num_tablas_vacias, orientation='h',marker_color="#00afb9"),
    #go.Bar(name='Tablas No Vacías', y=sistemas, x=num_tablas_no_vacias, orientation='h', marker_color='#0081a7')
    ])


    # Añadir etiquetas y título
    fig.update_layout(
        margin=dict(t=20, b=0, l=10, r=0),
        #title='Tablas vacías por sistema'.upper(),
        yaxis_title='Sistemas',
        xaxis_title='Tablas vacias',
        barmode='stack',
        height=700,
        #width=1200,
        #paper_bgcolor="#F3F2F2"
        )

    # Añadir los valores en las barras
    """for i in range(len(sistemas)):
        fig.add_annotation(
            x=num_tablas[i],
            y=sistemas[i],
            text=f'{int(num_tablas_vacias[i])} | {int(num_tablas[i])}<br> {porcentaje_tablas_vacias[i]}%',
            #showarrow=False,
            font=dict(size=14,
                    family="sans-serif",  # Tipo de fuente
                    color="black",  # Color del texto
                    weight="bold"
                    ),
            xanchor='left'
        )"""
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
    
#GRAFICO VALORES NULOS 
@app.callback(
    Output('dispersion-chart', 'figure'),
    [Input('bar-chart-sistema-tabs', 'active_tab')]
)
def update_graph_vall_null(select_system):
    colors = ["#0a2239","#336c81","#1d84b5","#00eaff","#00ffcc",'#00ff48', '#ff8009', '#f07167']
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
        #height=650,        #alto
        #width=600,        #ancho total
        showlegend=True,       
        title={
            'text': f'DISPERSION DE VALORES NULOS POR TABLA EN EL SISTEMA: {filtred}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 21,
                'family': 'sans-serif',
                 'color': '#A0A0A0'}
        },
        yaxis_title='Valores nulos',
        xaxis_title='Tablas'
    )
    return fig

def insertar_salto_linea(texto, longitud_max):
    if len(texto) > longitud_max:
        texto = texto[:longitud_max] + '\n' + texto[longitud_max:]
    return texto

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
    q1=f"{int(df_4['VALORES_NULOS'].describe()['25%'])} valores nulos"
    q2=f"{int(df_4['VALORES_NULOS'].describe()['50%'])} valores nulos"
    q3=f"{int(df_4['VALORES_NULOS'].describe()['75%'])} valores nulos"
    nombre_tabla_max = df_4['TABLE_NAME'][df_4['VALORES_NULOS'] == val_max].values[0]
    nombre_tabla_max = insertar_salto_linea(nombre_tabla_max, 18)
    nombre_tabla_min = df_4['TABLE_NAME'][df_4['VALORES_NULOS'] == val_min].values[0]
    nombre_tabla_min = insertar_salto_linea(nombre_tabla_min, 18)
    
    tabla_descripcion = {
        'Descripción': [
            f'Numero maximo de valores nulos en {filtred}',
            f'Numero minimo de valores nulos en {filtred}',
            f'Tabla con mayor numero de valores nulos en {filtred}',
            f'Tabla con menor numero de valores nulos en {filtred}',
            f'El 25% de las tablas en {filtred} tiene',
            f'El 50% de las tablas en {filtred} tiene',
            f'El 75% de las tablas en {filtred} tiene'
        ],
        'Valor': [
            "{:,.1f}".format(val_max),
            "{:,.1f}".format(val_min),
            nombre_tabla_max,
            nombre_tabla_min,
            q1,
            q2,
            q3
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

@app.callback(
    Output('name_system', 'children'),
    [Input('bar-chart-sistema-tabs', 'active_tab')]
)
def update_name(active_tab):
    return  names['NOMBRE COMPLETO'][names['ACRONIMO'] == active_tab].values[0]

@app.callback(
    Output('card_info', 'children'),
    [Input('bar-chart-sistema-tabs', 'active_tab')]
)
def cards_null(active_tab):
    df_5 = pd.concat(list_df_table, ignore_index=True)
    filtred = active_tab  # Extraer el nombre del sistema del tab_id
    media = df_5['VALORES_NULOS'][df_5['SISTEMA']==filtred].mean()
    
    card_1 = dbc.Card(
        dbc.CardBody(
            [
                html.P('NUMERO DE TABLAS', style={'font-size': '16px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P(f"{int(df['NUMERO TOTAL DE TABLAS'][df['SISTEMA'] == filtred].values[0])}", style={'font-size': '30px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': '220px',
            'height': '100%',
            'background-color': '#4d59ff',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_2 = dbc.Card(
        dbc.CardBody(
            [
                html.P('PROMEDIO DE VAL NULL', style={'font-size': '16px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P("{:,.1f}".format(media), style={'font-size': '30px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': '220px',
            'height': '100%',
            'background-color': '#0569ff',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    return [card_1, card_2]

# NUMERO DE REGISTROS
@app.callback(
    Output('name_system_2', 'children'),
    [Input('bar-chart-sistema-tabs_2', 'active_tab')]
)
def update_name_2(active_tab):
    return  names['NOMBRE COMPLETO'][names['ACRONIMO'] == active_tab].values[0]

@app.callback(
    Output('dispersion-num-registros', 'figure'),
    [Input('bar-chart-sistema-tabs_2', 'active_tab')]
)
def upgrade_graph_num_reg(active_tab):
    colors = ["#0a2239","#336c81","#1d84b5","#00eaff","#00ffcc",'#00ff48', '#ff8009', '#f07167']
    filtred = active_tab
    df_filtred_registro = df_registros[df_registros['SISTEMA']==filtred]
    sizeref = 2. * max(df_filtred_registro['bubble_size']) / (50**2)
     # Crear el gráfico usando Plotly
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(len(df_filtred_registro))),  # Ajustar el eje X a la longitud de los datos
        y=df_filtred_registro['NUM_ROWS'],
        name=filtred,
        text=df_filtred_registro['text'],
        marker=dict(size=df_filtred_registro['bubble_size'],
                        color=colors[random.randint(0,len(colors)-1)])
        ))
    fig2.update_traces(mode='markers', marker=dict(sizemode='area', sizeref=sizeref, line_width=2))
    # Ajustar el diseño
    fig2.update_layout(
        #height=650,        #alto
        #width=1300,        #ancho total
        showlegend=True, 
        title={
            'text': f'DISPERSION DE NUMERO DE REGISTROS POR TABLA <br> EN EL SISTEMA {filtred}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 22,
                'family': 'sans-serif',
                 'color': '#A0A0A0'}
        },
        yaxis_title='Numeros de registros',
        xaxis_title='Tablas'
    )
    # Mostrar el gráfico
    return fig2

@app.callback(
    Output('card_info_registro', 'children'),
    [Input('bar-chart-sistema-tabs_2', 'active_tab')]
)
def cards_num_registro(active_tab):
    filtred = active_tab  # Extraer el nombre del sistema del tab_id
    media = df_registros['NUM_ROWS'][df_registros['SISTEMA']==filtred].mean()
    
    card_3 = dbc.Card(
        dbc.CardBody(
            [
                html.P('NUMERO DE TABLAS', style={'font-size': '18px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P(f"{int(df['NUMERO TOTAL DE TABLAS'][df['SISTEMA'] == filtred].values[0])}", 
                       style={'font-size': '50px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '100%',
            'background-color': '#4d59ff',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_4 = dbc.Card(
        dbc.CardBody(
            [
                html.P('PROMEDIO DE REGISTROS', style={'font-size': '18px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P("{:,.1f}".format(media), style={'font-size': '30px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '100%',
            'background-color': '#0569ff',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    return [card_3, card_4]


@app.callback(
    Output('table-container_register', 'children'),
    [Input('bar-chart-sistema-tabs_2', 'active_tab')]
)
def update_table_2(active_tab):
    filtred = active_tab # Extraer el nombre del sistema del tab_id
    std = df_registros['NUM_ROWS'][df_registros['SISTEMA']==filtred].describe()['std']
    min = df_registros['NUM_ROWS'][df_registros['SISTEMA']==filtred].describe()['min']
    q1 = f"{str(int(df_registros['NUM_ROWS'][df_registros['SISTEMA']==filtred].describe()['25%']))} registros"
    q2 = f"{str(int(df_registros['NUM_ROWS'][df_registros['SISTEMA']==filtred].describe()['50%']))} registros"
    q3 = f"{str(int(df_registros['NUM_ROWS'][df_registros['SISTEMA']==filtred].describe()['75%']))} registros"
    max = df_registros['NUM_ROWS'][df_registros['SISTEMA']==filtred].describe()['max']
    nombre_tabla_min = df_registros[['TABLE_NAME','NUM_ROWS']][df_registros['SISTEMA']==filtred].min().values[0]
    nombre_tabla_max = df_registros[['TABLE_NAME','NUM_ROWS']][df_registros['SISTEMA']==filtred].max().values[0]
    
    tabla_descripcion = {
        'Descripción': [
            f'Numero maximo de registros en {filtred}',
            f'Numero minimo de registros en {filtred}',
            f'Tabla con mayor numero de registros en {filtred}',
            f'Tabla con menor numero de registros en {filtred}',
            f'Desviacion estandar de registros en {filtred}',
            f'El 25% de las tablas en {filtred} tiene',
            f'El 50% de las tablas en {filtred} tiene',
            f'El 75% de las tablas en {filtred} tiene'
        ],
        'Valor': [
            "{:,.1f}".format(max),
            "{:,.1f}".format(min),
            nombre_tabla_max,
            nombre_tabla_min,
             "{:,.1f}".format(std),
            q1,
            q2,
            q3
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

# NUMERO DE REGISTROS VALORES UNICOS

@app.callback(
    Output('dispersion-num-registros_unicos', 'figure'),
    [Input('bar-chart-sistema-tabs_3', 'active_tab')]
)
def upgrade_graph_num_registro_unique(select_system_2):
    colors = ["#0a2239","#336c81","#1d84b5","#00eaff","#00ffcc",'#00ff48', '#ff8009', '#f07167']
    filtred = select_system_2
    df_filtred_distinct=df_registros_unicos[df_registros_unicos['SISTEMA'] == filtred]
    sizeref = 2. * max(df_filtred_distinct['bubble_size_num_distinct']) / (50**2)

    # Crear el gráfico usando Plotly
    fig5 = go.Figure()

    fig5.add_trace(go.Scatter(
        x=list(range(len(df_filtred_distinct))),  # Ajustar el eje X a la longitud de los datos
        y=df_filtred_distinct['NUM_DISTINCT'],
        name=filtred,
        text=df_filtred_distinct['text_num_distinct'],
        marker=dict(size=df_filtred_distinct['bubble_size_num_distinct'],color=colors[random.randint(0,len(colors)-1)])
        ))

    fig5.update_traces(mode='markers', marker=dict(sizemode='area', sizeref=sizeref, line_width=2))


    # Ajustar el diseño
    fig5.update_layout(
        #height=700,        #alto
        #width=1300,        #ancho total
        showlegend=True,       
        title={
            'text': f'Dispersion del numero de registros <br> unicos en los sistemas de clientes',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 22,
                'family': 'sans-serif',
                 'color': '#A0A0A0'}
        },
        yaxis_title='Valores unicos',
        xaxis_title='Tablas')
    return fig5

@app.callback(
    Output('name_system_3', 'children'),
    [Input('bar-chart-sistema-tabs_3', 'active_tab')]
)
def update_name_3(select_system_2):
    return  names['NOMBRE COMPLETO'][names['ACRONIMO'] == select_system_2].values[0]


@app.callback(
    Output('card_info_registro_unicos','children'),
    [Input('bar-chart-sistema-tabs_3', 'active_tab')]
)
def cards_num_registro_unique(select_system_2):
    filtred = select_system_2  # Extraer el nombre del sistema del tab_id
    media = df_registros_unicos['NUM_DISTINCT'][df_registros_unicos['SISTEMA']==filtred].mean()
    
    card_10 = dbc.Card(
        dbc.CardBody(
            [
                html.P('NUMERO DE TABLAS', style={'font-size': '18px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P(f"{int(df['NUMERO TOTAL DE TABLAS'][df['SISTEMA'] == filtred].values[0])}", 
                       style={'font-size': '50px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '100%',
            'background-color': '#4d59ff',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_11 = dbc.Card(
        dbc.CardBody(
            [
                html.P('PROMEDIO DE REGISTROS UNICOS', style={'font-size': '18px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P("{:,.1f}".format(media), style={'font-size': '30px', 'color': 'white', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '100%',
            'background-color': '#0569ff',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    return [card_10, card_11]


@app.callback(
    Output('table-container_register_unicos','children'),
    [Input('bar-chart-sistema-tabs_3', 'active_tab')]
)
def update_table_3(select_system_2):
    filtred = select_system_2 # Extraer el nombre del sistema del tab_id
    std = df_registros_unicos['NUM_DISTINCT'][df_registros_unicos['SISTEMA']==filtred].describe()['std']
    min = df_registros_unicos['NUM_DISTINCT'][df_registros_unicos['SISTEMA']==filtred].describe()['min']
    q1 = f"{str(int(df_registros_unicos['NUM_DISTINCT'][df_registros_unicos['SISTEMA']==filtred].describe()['25%']))} registros unicos"
    q2 = f"{str(int(df_registros_unicos['NUM_DISTINCT'][df_registros_unicos['SISTEMA']==filtred].describe()['50%']))} registros unicos"
    q3 = f"{str(int(df_registros_unicos['NUM_DISTINCT'][df_registros_unicos['SISTEMA']==filtred].describe()['75%']))} registros unicos"
    max = df_registros_unicos['NUM_DISTINCT'][df_registros_unicos['SISTEMA']==filtred].describe()['max']
    nombre_tabla_min = df_registros_unicos[['TABLE_NAME','NUM_DISTINCT']][df_registros_unicos['SISTEMA']==filtred].min().values[0]
    nombre_tabla_max = df_registros_unicos[['TABLE_NAME','NUM_DISTINCT']][df_registros_unicos['SISTEMA']==filtred].max().values[0]
    
    tabla_descripcion = {
        'Descripción': [
            f'Numero maximo de registros en {filtred}',
            f'Numero minimo de registros en {filtred}',
            f'Tabla con mayor numero de registros en {filtred}',
            f'Tabla con menor numero de registros en {filtred}',
            f'Desviacion estandar de registros en {filtred}',
            f'El 25% de las columnas en {filtred} tiene',
            f'El 50% de las columnas en {filtred} tiene',
            f'El 75% de las columnas en {filtred} tiene'
        ],
        'Valor': [
            "{:,.1f}".format(max),
            "{:,.1f}".format(min),
            nombre_tabla_max,
            nombre_tabla_min,
             "{:,.1f}".format(std),
            q1,
            q2,
            q3
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


# ALMACENAMIENTO
@app.callback(
    Output('treemap_file','figure'),
    Input('treemap_file', 'id') 
)
def treemap_storage(_):
    fig = px.treemap(df_storage, path=[px.Constant("Data Lake"),'text'],
                 values='GB_log', 
                 color='gb',
                 color_continuous_scale='blues')
                 #width=700,
                 #height=550)

    fig.update_traces(
        marker=dict(cornerradius=10),
        #customdata=df['text'],
        hovertemplate= '%{label}',
        textposition='middle center'
    )

    fig.update_layout(
        margin = dict(t=5, l=5, r=5, b=5))
    return fig 

@app.callback(
    Output('bar_storage','figure'),
    Input('bar_storage', 'id') 
)
def bar_storage(_):
    # Crear la gráfica de barras
    df_storage['Porcentaje']=df_storage['gb'].apply(lambda x: x/df_storage['gb'].sum() * 100)
    df_storage["text"] = df_storage.apply(lambda row: f"{row['SISTEMAS']} <br> GB: {round(row['gb'],2)} <br> MB: {round(row['mb'],2)} <br> BYTES: {round(row['bytes'],2)} <br> {round(row['Porcentaje'],3)}%", axis=1)

    fig = go.Figure(data=[
        go.Bar(x=df_storage['SISTEMAS'].tolist(),
               y=df_storage['GB_log'].tolist(),
               orientation='v',
               name='ALMACENAMIENTO',
               marker_color='steelblue',
               customdata=df_storage[['text']],
               hovertemplate='<b>%{customdata[0]}</b> <extra></extra>'),
    ])


    # Cambiar el diseño de la gráfica
    fig.update_layout(
        barmode='overlay',
        showlegend=False,       
        title={
            'text': f'Comparación del Almacenamiento <br> con Escala Logarítmica',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 21,
                'family': 'sans-serif',
                 'color': '#A0A0A0'}
        },
        xaxis_title='ALMACENAMIENTO',
        yaxis_title='SISTEMAS',
        legend_title='Tipo de Tabla'
    )
    return fig 

@app.callback(
    Output('kpis_storage', 'children'),
    [Input('kpis_storage', 'id')]
)
def kpis_storage(_):
    storage =  round(df_storage['gb'].sum(),2)
    
    card_5 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Datos a Migrar', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P(f'{round(storage)}GB', 
                       style={'font-size': '50px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_6 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Numero de tablas a Migrar', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P(' ', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P("2998", 
                       style={'font-size': '50px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_8 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Progreso: ', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P('3%', 
                       style={'font-size': '50px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_7 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Almacenamiento necesario', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P(' ', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P('2TB', 
                       style={'font-size': '50px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    return card_5,card_6,card_7,card_8

# DISTRISTIBUCION DE TIPO DE DATO:

@app.callback(
    Output('bar_type_data','figure'),
    Input('bar_type_data', 'id') 
)
def bar_type_data(_):
    data_types_description = {
    "NUMBER": "Tipo de dato numérico que puede almacenar números enteros y decimales.",
    "VARCHAR2": "Tipo de dato de cadena de caracteres de longitud variable.",
    "DATE": "Tipo de dato que almacena fechas.",
    "CHAR": "Tipo de dato de cadena de caracteres de longitud fija.",
    "NCLOB": "Tipo de dato que almacena grandes objetos de texto en Unicode.",
    "UNDEFINED": "Tipo de dato no definido o desconocido.",
    "TIMESTAMP(6)": "ipo de dato que almacena una fecha y hora con precisión de hasta 6 decimales en los segundos.",
    "BLOB": "Tipo de dato que almacena grandes objetos binarios.",
    "CLOB": "Tipo de dato que almacena grandes objetos de texto.",
    "LONG": "Tipo de dato que almacena grandes cadenas de caracteres.",
    "RAW": "Tipo de dato que almacena datos binarios en bruto.",
    "ROWID": "Tipo de dato que almacena el identificador de fila único de una tabla.",
    "INTERVAL DAY(2) TO SECOND(6)": "Tipo de dato que almacena un intervalo de tiempo en días, horas, minutos y segundos con precisión de hasta 6 decimales en los segundos.",
    "FLOAT": "Tipo de dato numérico que almacena números en coma flotante.",
    "INTERVAL DAY(3) TO SECOND(6)": "Tipo de dato que almacena un intervalo de tiempo en días, horas, minutos y segundos con precisión de hasta 6 decimales en los segundos.",
    "NVARCHAR2": "Tipo de dato de cadena de caracteres de longitud variable en Unicode.",
    "LONG RAW": "Tipo de dato que almacena grandes objetos binarios en bruto."
    }
    data_counts = df_registros_unicos['DATA_TYPE'].value_counts()
    data_counts_df = data_counts.reset_index()
    data_counts_df.columns = ['DATA_TYPE', 'count']
    data_counts_df['count_log'] =np.log1p(data_counts_df['count'])
    total_columnas=data_counts_df['count'].sum()
    data_counts_df['text'] = data_counts_df.apply(lambda row: f"Tipo de dato: {row['DATA_TYPE']} <br> Cantidad: {row['count']} <br> Total de columna: {total_columnas} <br> Representa: {round((row['count']/total_columnas)*100,4)}% <br> Descripcion: {data_types_description[row['DATA_TYPE']]}", axis=1)
    fig = px.bar(data_counts_df, x='DATA_TYPE', y='count_log')

    fig.update_traces(
    customdata=data_counts_df[['text']],
    hovertemplate= '%{customdata[0]}'
    )
    fig.update_layout(
        #height=700,        #alto
        #width=1300,        #ancho total
        showlegend=True,       
        title={
            'text': f'DISTRISTIBUCION DE TIPO DE DATO EN <br> LOS SISTEMA DE CLIENTES EN ESCALA LOGARITMICA',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 22,
                'family': 'sans-serif',
                 'color': '#A0A0A0'}
        },
        yaxis_title='CONTEO',
        xaxis_title='TIPO DE DATO')
    
    return fig 


@app.callback(
    Output('kpis_type_data', 'children'),
    [Input('kpis_type_data', 'id')]
)
def kpis_type_data(_):
    storage =  round(df_storage['gb'].sum(),2)
    
    card_12 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Numero de columnas', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P(f'{len(df_registros_unicos)}', 
                       style={'font-size': '50px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_13 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Tipo de dato predominate', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P(' ', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P("NUMBER", 
                       style={'font-size': '30px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_14 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Frecuencia Absoluta DMC', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P('19454', 
                       style={'font-size': '50px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_15 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Frecuencia relativa DMC', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                #html.P(' ', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P('42.68%', 
                       style={'font-size': '50px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    card_16 = dbc.Card(
        dbc.CardBody(
            [
                html.P('Tipo de dato atipico', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                #html.P(' ', style={'font-size': '18px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.P('UNDEFINED', 
                       style={'font-size': '30px', 'color': 'black', 'font-weight': 'bold', 'textAlign': 'center'})
            ]
        ),
        style={
            'display': 'flex',
            'width': "250px",
            'height': '95%',
            'background-color': 'white',
            'margin': '5px',
            'border-radius': '15%'
        }
    )
    
    return card_12,card_13,card_14,card_15,card_16

if __name__ == '__main__':
    app.run_server(debug=True, port = 8051)