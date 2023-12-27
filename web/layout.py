import dash
from dash import dcc, html
import dash_daq as daq

# Initialize Dash application
app = dash.Dash(__name__)

data = {
    'signaturesCOSMIC.csv': [x for x in range(1, 31) ],
    'signaturesProfiler.csv': [x for x in range(0, 64)],
    'COSMIC_v1_SBS_GRCh37.txt': [1, 2, 3, 4, 5, 6, 7, 9, 15],
    'COSMIC_v2_SBS_GRCh37.txt': [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30],
    'COSMIC_v3.1_SBS_GRCh37.txt': [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30],
}
organs = [
    "Breast", "Ovary", "Kidney", "Colorectal", "Bone_SoftTissue",
    "Lung", "Uterus", "CNS", "Prostate", "Bladder", "Skin",
    "Stomach", "NET", "Pancreas", "Biliary", "Liver", "Lymphoid",
    "Myeloid", "Oral_Oropharyngeal", "Esophagus", "Head_neck"
]
# Application layout
app.layout = html.Div([
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '300px',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        dcc.Store(id='session', storage_type='session', data=None),
        daq.BooleanSwitch(
                id='dropdown-switch',
                on=True,
                label='Choose Organ',
                labelPosition='top',
        ),
        dcc.Dropdown(
            id='organ-dropdown',
            options=[{'label': organ, 'value': organ} for organ in organs],
            value='Breast',
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
        ),
        html.Div(id='upload-message',
                 style={
                     'width': '50%',
                     'height': '60px',
                     'lineHeight': '60px',
                     'borderWidth': '1px',
                     'borderStyle': 'line',
                     'borderRadius': '5px',
                     'textAlign': 'center',
                     'margin': '10px'
                 },
        ),
    ], style={'display': 'flex', 'justifyContent': 'center'}),


    dcc.Dropdown(
        id='patient-dropdown',
        options=[{'label': 'None', 'value': 'None'}],
        value=None
    ),
    html.Div([
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'signaturesProfiler', 'value': 'signaturesProfiler.csv'},
                {'label': 'signaturesCOSMIC', 'value': 'signaturesCOSMIC.csv'},
                {'label': 'COSMIC_v1_SBS_GRCh37', 'value': 'COSMIC_v1_SBS_GRCh37.txt'},
                {'label': 'COSMIC_v2_SBS_GRCh37.txt', 'value': 'COSMIC_v2_SBS_GRCh37.txt'},
                {'label': 'COSMIC_v3.1_SBS_GRCh37.txt', 'value': 'COSMIC_v3.1_SBS_GRCh37.txt'},
                {'label': 'COSMIC_v3.4_SBS_GRCh37.txt', 'value': 'COSMIC_v3.4_SBS_GRCh37.txt'},
            ],
            disabled=True,
            value='signaturesProfiler.csv'
        ),
        dcc.Dropdown(
            id='signatures-dropdown',
            options=[{'label': k, 'value': k} for k in data.keys()],
            multi=True,
            value=[k for k in data['signaturesProfiler.csv']],
        ),
    ], style={'padding': '10px'}),

    html.Div([
        dcc.Slider(
            id='fold_size-slider',
            min=0,
            max=10,  # Example maximum value, adjust as needed
            step=1,
            value=4,  # Default value
        )
    ], style={'padding': '20px'}),
    html.Div([
        dcc.Input(id='input-R', type='number', value=10, style={'marginRight': '10px'}),
        dcc.Input(id='input-mutation-count', type='number', value=0, style={'marginRight': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '10px'}),
    html.Button('Wyczyść dane', id='clear-button'),

    html.Div(id='slider-output-container'),
    dcc.Graph(id='bar-plot-crossvalid'),
    dcc.Graph(id='bar-plot-bootstrap'),
    dcc.Graph(id='bar-plot-modelselection'),
    dcc.Graph(id='bar-plot-forward_model')
])
