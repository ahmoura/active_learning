from pathlib import Path
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from pyhard.utils import reduce_dim

_my_path = Path(__file__).parent
_folder = 'overlap'
datadir = _my_path / 'data'
list_dir = [x.name for x in datadir.glob('**/*') if x.is_dir()]
list_dir.sort()

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, meta_tags=[
    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
])

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id='dataset',
                    options=[{'label': i, 'value': i} for i in list_dir],
                    value='overlap',
                    clearable=False
                ),
            ]
        )
    ],
    body=True
)

app.layout = dbc.Container(
    [
        html.H1("Instance Hardness"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=2, width=2),
                dbc.Col(dcc.Graph(id='g1')),
                dbc.Col(dcc.Graph(id='g2')),
            ],
        ),
        dbc.Row(html.Div(id='hidden', style={'display': 'none'}))
    ],
    fluid=True
)


# app.layout = html.Div([
#     #     dcc.Dropdown(
#     #         id='dataset',
#     #         options=[{'label': i, 'value': i} for i in list_dir],
#     #         value='overlap',
#     #         clearable=False
#     #     )
#     # ], style={'width': '19%', 'display': 'inline-block'}),
#     # html.Div([
#     #     dcc.Graph(id='g1', responsive=True)
#     # ], style={'width': '40%', 'display': 'inline-block', 'padding': '0 20'}),
#     # html.Div([
#     #     dcc.Graph(id='g2', responsive=True)
#     # ], style={'width': '40%', 'display': 'inline-block'}),
#
#     dbc.Row(
#         [
#             dbc.Col(html.Div(dcc.Dropdown(
#                 id='dataset',
#                 options=[{'label': i, 'value': i} for i in list_dir],
#                 value='overlap',
#                 clearable=False
#             ))),
#             dbc.Col(html.Div(dcc.Graph(id='g1'))),
#             dbc.Col(html.Div(dcc.Graph(id='g2'))),
#         ]
#     ),
#     dbc.Row(html.Div(id='hidden', style={'display': 'none'}))
# ])


@app.callback(Output('hidden', 'children'), Input('dataset', 'value'))
def load_data(folder):
    dim_method = 'PCA'
    folder = datadir / folder
    dataset = pd.read_csv(folder / 'data.csv')

    if len(dataset.columns) > 3:
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        X_embedded = reduce_dim(X, y, method=dim_method)
        df = pd.DataFrame(X_embedded, columns=['V1', 'V2'], index=X.index)
        dataset = pd.concat([df, y], axis=1)

    df_metadata = pd.read_csv(folder / 'metadata.csv', index_col='instances')
    df_is = pd.read_csv(folder / 'coordinates.csv', index_col='Row')
    df_is.index.name = 'instances'

    dataset.index = df_metadata.index
    df_data = df_is.join(dataset)
    df_data = df_data.join(df_metadata)
    df_data.index = df_data.index.map(str)

    data_cols = dataset.columns.to_list()
    is_cols = df_is.columns.to_list()
    keys_dict = dict(is_kdims=is_cols[0:2],
                     data_dims=dataset.columns.to_list(),
                     data_kdims=data_cols[0:2],
                     class_label=data_cols[2],
                     meta_dims=df_metadata.columns.to_list())

    return df_data.to_json(date_format='iso', orient='split')


@app.callback(Output('g1', 'figure'),
              Output('g2', 'figure'),
              Input('hidden', 'children')
              )
def update_graph(jsonified_cleaned_data):
    # more generally, this line would be
    # json.loads(jsonified_cleaned_data)
    df = pd.read_json(jsonified_cleaned_data, orient='split')

    fig1 = px.scatter(df, x='V1', y='V2', color='class')
    fig1.update_layout(autosize=True)
    fig2 = px.scatter(df, x='z_1', y='z_2', color='class')
    # fig2.update_layout(width=500, height=400)
    return [fig1, fig2]


# @app.callback(
#     Output('g1', 'figure'),
#     Output('g2', 'figure'),
#     Input('g1', 'selectedData')
# )
# def callback(selection):
#     print(selection)
#
#     selectedpoints = df.index
#     if selection and selection['points']:
#         selectedpoints = np.intersect1d(selectedpoints, [p['pointIndex'] for p in selection['points']])
#
#     print(selectedpoints)
#
#     fig = px.scatter(df, x=df['Col 1'], y=df['Col 2'], text=df.index)
#
#     fig.update_traces(selectedpoints=selectedpoints,
#                       customdata=df.index,
#                       mode='markers+text', marker={'color': 'rgba(0, 116, 217, 0.7)', 'size': 20},
#                       unselected={'marker': {'opacity': 0.3}, 'textfont': {'color': 'rgba(0, 0, 0, 0)'}})
#
#     fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)
#
#     fig2 = px.scatter(df.loc[selectedpoints, :], x='Col 1', y='Col 2')
#
#     fig2.update_traces(
#         customdata=selectedpoints,
#         mode='markers+text', marker={'color': 'rgba(0, 116, 217, 0.7)', 'size': 20},
#     )
#
#     fig2.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)
#
#     return [fig, fig2]


if __name__ == '__main__':
    app.run_server(debug=True)
