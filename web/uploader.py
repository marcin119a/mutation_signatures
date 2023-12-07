import base64
import numpy as np
from dash import html
import io
import re

# Function to parse CSV file content
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            data = np.genfromtxt(io.StringIO(decoded.decode('utf-8')), delimiter=',', skip_header=1, dtype=float)[:, 1:]
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

def load_signatures(filename, organ=False):
    delimiter = ','
    if organ:
        file_path = f"../data/signatures_organ/latest/{filename}_Signature.csv"
    else:
        delimiter = '\t' if '.txt' in filename else ','
        file_path = f'../data/{filename}'
    signatures = np.genfromtxt(file_path, delimiter=delimiter, skip_header=1)

    return np.delete(signatures, 0, axis=1)

def load_names(filename):
    file_path = f"../data/signatures_organ/latest/{filename}_Signature.csv"
    signatures = np.genfromtxt(file_path, delimiter=',', max_rows=1, dtype='str')[1:]

    pattern = r'([^_]+)$'
    return [re.search(pattern, item).group() for item in signatures if re.search(pattern, item)]