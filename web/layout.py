import dash
from web.uploader import parse_contents, load_signatures
from estimates_exposures import bootstrapSigExposures, crossValidationSigExposures, findSigExposures
import numpy as np
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

# Initialize Dash application
app = dash.Dash(__name__)

# Application layout
app.layout = html.Div([
    html.Div([
        # Komponent Upload w pierwszym wierszu
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        )
    ], style={'display': 'flex', 'justifyContent': 'center'}),
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
        dcc.Input(id='input-mutation-count', type='number', value=1000, style={'marginRight': '10px'}),
        dcc.Input(id='patient', type='text', value='PD24196'),
    ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '10px'}),

    html.Div([
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'signaturesCOSMIC', 'value': 'signaturesCOSMIC.csv'},
                {'label': 'COSMIC_v1_SBS_GRCh37', 'value': 'COSMIC_v1_SBS_GRCh37.txt'},
                {'label': 'COSMIC_v2_SBS_GRCh37.txt', 'value': 'COSMIC_v2_SBS_GRCh37.txt'},
                {'label': 'COSMIC_v3.1_SBS_GRCh37.txt', 'value': 'COSMIC_v3.1_SBS_GRCh37.txt'},
                {'label': 'COSMIC_v3.4_SBS_GRCh37.txt', 'value': 'COSMIC_v3.4_SBS_GRCh37.txt'},
            ],
            value='signaturesCOSMIC.csv'  # wartość domyślna
        )
    ], style={'padding': '10px'}),

    html.Div(id='slider-output-container'),
    dcc.Graph(id='bar-plot-crossvalid'),
    dcc.Graph(id='bar-plot-bootstrap')
])

# Callback to update the plot based on data and parameters
@app.callback(
    [Output('bar-plot-crossvalid', 'figure'), Output('bar-plot-bootstrap', 'figure')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('fold_size-slider', 'value'),
     Input('input-R', 'value'),
     Input('input-mutation-count', 'value'),
     Input('dropdown', 'value')],  # Dodanie listy rozwijanej jako Input
    [State('patient', 'value')]
)
def update_output(contents, filename, fold_size, R, mutation_count, dropdown_value, patient):
    if contents is not None:
        data, patients = parse_contents(contents, filename)
        column_index = np.where(patients == patient)[0]

        patient_column = data[:, column_index].squeeze()
        sigsBRCA = [x - 1 for x in [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]]
        signatures = load_signatures(dropdown_value)[:, sigsBRCA]

        exposures, errors = findSigExposures(patient_column.reshape(patient_column.shape[0], 1), signatures)

        exposures_cv, errors_cv = crossValidationSigExposures(patient_column, signatures, fold_size)

        fig_cross = px.strip(x=range(exposures.shape[0]), y=exposures.squeeze(), stripmode='overlay')

        for i in range(exposures_cv.shape[0]):
            fig_cross.add_trace(go.Box(
                y=exposures_cv[i, :],
                name=f'Sig {sigsBRCA[i] + 1}'))

        fig_cross.update_layout(
            title=f'Cross valid for {patient}',
            xaxis_title='Sig',
            yaxis_title='Signature contribution'
        )

        exposures_bt, errors_bt = bootstrapSigExposures(patient_column, signatures, R, mutation_count)

        fig_bootstrap = px.strip(x=range(exposures.shape[0]), y=exposures.squeeze(), stripmode='overlay')

        for i in range(exposures_bt.shape[0]):
            fig_bootstrap.add_trace(go.Box(
                y=exposures_bt[i, :],
                name=f'Sig {sigsBRCA[i] + 1}'))

        fig_bootstrap.update_layout(
            title=f'Bootstrap for {patient}',
            xaxis_title='Sig',
            yaxis_title='Signature contribution'
        )

        return fig_cross, fig_bootstrap
    else:
        return dash.no_update

# Callback to display the value of the slider
@app.callback(
    Output('slider-output-container', 'children'),
    [Input('fold_size-slider', 'value'),
     Input('input-R', 'value'),
     Input('input-mutation-count', 'value')]
)
def update_output(value, R, mutation_count):
    return ' fold_size {} R: {}, mutation_count: {}'.format(value, R, mutation_count)

