import base64
import pandas as pd
from dash import  html
import io

# Function to parse CSV file content
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Case for CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return html.Div([
                'There was an error processing this file.'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df