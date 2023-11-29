import base64
import numpy as np
from dash import html
import io

# Function to parse CSV file content
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Case for CSV file
            data = np.genfromtxt(io.StringIO(decoded.decode('utf-8')), delimiter=',', skip_header=1)
            patients = np.genfromtxt(io.StringIO(decoded.decode('utf-8')), delimiter=',', max_rows=1, dtype=str)[1:]
            patients = np.char.strip(patients, '"')
        else:
            return html.Div([
                'There was an error processing this file.'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return data, patients

def load_signatures(filename):
    delimiter = '\t' if '.txt' in filename else ','
    filepath = f'../data/{filename}'
    signatures = np.genfromtxt(filepath, delimiter=delimiter, skip_header=1)

    return np.delete(signatures, 0, axis=1)