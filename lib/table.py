import pandas as pd


def format_pipe_table(*args, **kwargs):
    # Rows is a dictionary of keys
    df = pd.DataFrame(*args, **kwargs)
    ret = ''

    columns = ['Index'] + list(df.columns.values)

    # Header
    ret += '| ' + ' | '.join(columns) + ' |\n'
    ret += '| ' + ' | '.join(['---' for _ in columns]) + ' |\n'

    # Add values
    for index, row in df.iterrows():
        values = [index] + list(row)
        values = [str(v) for v in values]
        ret += '| ' + ' | '.join(values) + ' |\n'
    return ret