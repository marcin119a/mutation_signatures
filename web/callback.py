from web.layout import app, data, organs
import plotly.graph_objects as go
import plotly.express as px
from web.uploader import parse_contents, load_signatures, load_names
from estimates_exposures import bootstrapSigExposures, crossValidationSigExposures, findSigExposures
import numpy as np
from utils import is_wholenumber
from dash import Input, Output, State
import dash


@app.callback(
    [Output('signatures-dropdown', 'options'),
     Output('signatures-dropdown', 'value')],
    [Input('dropdown', 'value')]
)
def set_options(selected_category):
    return [{'label': f"Sig {i}", 'value': i} for i in data[selected_category]], [i for i in data[selected_category]]

@app.callback(
    [Output('session', 'data')],
    [Input('upload-data', 'contents'),
     Input('organ-dropdown', 'value')],
    [State('upload-data', 'filename')]
)
def update_output(contents, organ, filename):
    if contents is not None:
        data, patients = parse_contents(contents, filename)
        return [{'data': data, 'patients': patients, 'filename': filename, 'organ': organ}]
    else:
        return dash.no_update



# Callback to update the plot based on data and parameters
@app.callback(
    [Output('bar-plot-crossvalid', 'figure'),
     Output('bar-plot-bootstrap', 'figure'),
     Output('input-mutation-count', 'value'),
     ],
    [
     Input('fold_size-slider', 'value'),
     Input('input-R', 'value'),
     Input('input-mutation-count', 'value'),
     Input('patient-dropdown', 'value'),
     Input('session', 'data'),
     Input('signatures-dropdown', 'value'),
     Input('organ-dropdown', 'value')
     ],
    [State('dropdown', 'value'),
     State('dropdown-switch', 'on')]
)
def update_output(fold_size, R, mutation_count, patient, stored_data, signatures, organ, dropdown_value, boolean_on):
    if stored_data is not None and patient is not None:
        data, patients = np.array(stored_data['data']), np.array(stored_data['patients'])
        column_index = np.where(patients == patient)[0]

        patient_column = data[:, column_index].squeeze()

        if mutation_count == 0:
            if all(is_wholenumber(val) for val in patient_column):
                mutation_count = int(patient_column.sum())
        else:
            mutation_count = 1000

        if boolean_on:
            signatures = load_signatures(organ, organ=True)
            sigsBRCA = load_names(organ)
        else:
            sigsBRCA = [x - 1 for x in signatures]
            signatures = load_signatures(dropdown_value, organ=False)[:, sigsBRCA]

        exposures, errors = findSigExposures(patient_column.reshape(patient_column.shape[0], 1), signatures)

        exposures_cv, errors_cv = crossValidationSigExposures(patient_column, signatures, fold_size)

        fig_cross = px.strip(x=range(1, exposures.shape[0] + 1),
                             y=exposures.squeeze(),
                             stripmode='overlay')

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

        fig_bootstrap = px.strip(x=range(1, exposures.shape[0] + 1),
                             y=exposures.squeeze(),
                             stripmode='overlay')

        for i in range(exposures_bt.shape[0]):
            fig_bootstrap.add_trace(go.Box(
                y=exposures_bt[i, :],
                name=f'Sig {sigsBRCA[i] + 1}'))

        fig_bootstrap.update_layout(
            title=f'Bootstrap for {patient}',
            xaxis_title='Sig',
            yaxis_title='Signature contribution'
        )

        return fig_cross, fig_bootstrap, mutation_count
    else:
        return None, None, 0

# Callback to display the value of the slider
@app.callback(
    Output('slider-output-container', 'children'),
    [Input('fold_size-slider', 'value'),
     Input('input-R', 'value'),
     Input('input-mutation-count', 'value')]
)
def update_output(value, R, mutation_count):
    return ' fold_size {} R: {}, mutation_count: {}'.format(value, R, mutation_count)

from dash import html
@app.callback(
    Output('upload-message', 'children'),
    Output('patient-dropdown', 'options'),
    Output('patient-dropdown', 'value'),
    [Input('session', 'data')]
)
def update_message(data):
    if data is not None:
        return html.Div(f'File {data["filename"]} has been uploaded.'), [{'label': patient, 'value': patient} for patient in data['patients']], data['patients'][0]
    return '', [{'label': 'None', 'value': 'None'}], None

@app.callback(
    [Output('organ-dropdown', 'style'),
     Output('dropdown', 'disabled')],
    [Input('dropdown-switch', 'on')]
)
def toggle_dropdown(on_switch):
    if on_switch:
        return {
                'width': '50%',
                'height': '60px',
                'display': 'inline-block',
            }, False
    else:
        return {'display': 'none'}, True
