import json

from lazypeon import Path
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_html_components as html

# Preparação da figura
def buildFig(x, y, z, func):
    fig = go.Figure(data=go.Isosurface(
        x = x.flatten(),
        y = y.flatten(),
        z = z.flatten(),
        value = func(x,y,z).flatten(),
        isomin = 0,
        isomax = 0,
        opacity = 0.25,
        colorscale ='hot',
        showscale = False
        ))
    
    fig.update_layout(
        margin=dict(t=0, l=0, b=0), #tight layout
        )
    
    return fig

# Função de uma esfera
def f(x, y, z):
    f = x**2 + y**2 + z**2 - 1
    return f

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

x, y, z = np.mgrid[-2:2:50j, -2:2:50j, -2:2.:50j]
fig = buildFig(x, y, z, f)

app.layout = html.Div([
    html.Div([
        html.H2("Coordenadas selecionadas:"),
        html.Table([
            html.Tr([
                html.Td("x:"),
                dcc.Input(id="x-coord", value=0, type="text"),
                html.Td("y:"),
                dcc.Input(id="y-coord", value=0, type="text"),
                html.Td("z:"),
                dcc.Input(id="z-coord", value=0, type="text")
            ])
        ]),

        html.Div([
            html.H4(id= "A-coords", children="[0,0,0]"),
            html.Button("Atualizar A", id="A-button", n_clicks=0),
            html.H4(id= "B-coords", children="[0,0,0]"),
            html.Button("Atualizar B", id="B-button", n_clicks=0)
        ])
    ]),
    dcc.Graph(
        id='interactive-graph',
        figure=fig
    ),
    html.Div([
        html.Button("Começar", id="run-button", n_clicks=0)
    ]),
    html.Div(style={"dysplay":"flex"},
    children=[
        html.Div(style={"width": "50%", "float": "left"},
            children=[
            html.H4("Número de partículas:"),
            dcc.Input(id="particles-input", value=20, type="text")
        ]),
        html.Div(style={"width": "50%", "float": "left"},
        children= [
            html.H4("Número de pontos:"),
            dcc.Input(id="point-input", value=5, type="text")
        ]),
        html.Div(style={"width": "50%", "float": "left"},
        children= [
            html.H4("Gerações:"),
            dcc.Input(id="generations-input", value=1000, type="text")
        ]),
        html.Div(style={"width": "50%", "float": "left"},
        children= [
            html.H4("Taxa de mutação:"),
            dcc.Input(id="mutation-input", value=0.2, type="text")
        ])
    ]),
    html.Div([
        html.H2("Superfície (definida implicitamente):"),
        dcc.Input(id="isosurf-input", value='x**2 + y**2 + z**2 - 1', type="text"),
        html.Button("Atualizar", id="plot-button", n_clicks=0)
    ])
])

@app.callback(
    Output('interactive-graph', 'figure'),
    [Input('run-button', 'n_clicks'),
     Input('plot-button', 'n_clicks')],
    [State('generations-input', 'value'),
     State('point-input', 'value'),
     State('particles-input', 'value'),
     State('mutation-input', 'value'),
     State('isosurf-input', 'value'),
     State('A-coords', 'children'),
     State('B-coords', 'children')],
    prevent_initial_call=True)
def graph_callback(run_clicks, update_clicks, gens, points, particles, rate, function, A, B):
    ctx = dash.callback_context
    trig = ctx.triggered[0]['prop_id'].split('.')[0]

    x, y, z = np.mgrid[-2:2:50j, -2:2:50j, -2:2.:50j]
    f = eval("lambda x,y,z:"+function)
    figure = buildFig(x, y, z, f)

    if (trig == "run-button"):
        gens = int(gens)
        points = int(points)
        particles = int(particles)
        rate = float(rate)
        path = Path(max_generations=gens, mutation_rate=rate, crossover_p=0.7, points=points, n_particles=particles, n_best=2)
        A = eval(A)
        B = eval(B)
        fitness = path.fit(f, A, B)
        print(fitness)

        
        p = path.gbest.state
        figure.add_trace(go.Scatter3d(
        x = p[:,0],
        y = p[:,1],
        z = p[:,2],
        line=dict(
            color='green',
            width=2
            )
        ))
    else:
        print(function)

    return figure


@app.callback(
    Output('A-coords', 'children'),
    [Input('A-button', 'n_clicks')],
    [State('x-coord', 'value'),
    State('y-coord', 'value'),
    State('z-coord', 'value')],
    prevent_initial_call=True)
def update_A(n, x, y, z):
    A = [x, y, z]
    return f"[{x}, {y}, {z}]"

@app.callback(
    Output('B-coords', 'children'),
    [Input('B-button', 'n_clicks')],
    [State('x-coord', 'value'),
    State('y-coord', 'value'),
    State('z-coord', 'value')],
    prevent_initial_call=True)
def update_B(n, x, y, z):
    B = [x, y, z]
    return f"[{x}, {y}, {z}]"
    
@app.callback(
    [Output('x-coord', 'value'),
    Output('y-coord', 'value'),
    Output('z-coord', 'value')],
    Input('interactive-graph', 'clickData'),
    prevent_initial_call=True)
def display_click_data(clickData):
    print(clickData['points'][0]['x'], clickData['points'][0]['y'], clickData['points'][0]['z'])
    return clickData['points'][0]['x'], clickData['points'][0]['y'], clickData['points'][0]['z']


if __name__ == '__main__':
    app.run_server(debug=True)
