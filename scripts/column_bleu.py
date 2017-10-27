"""
column_bleu.py computes the BLEU between two columns. 
"""

import pandas

from metrics.bleu import moses_multi_bleu

def main(src='data/simple_qa/test.tsv',
         column_a='Question EN',
         column_b='Question EN DeepL'):
    """ Computes the BLEU between two columns """
    data = pandas.read_table(src)
    assert column_a in data.columns.values
    assert column_b in data.columns.values
    data = data[~data[column_a].isnull()]
    data = data[~data[column_b].isnull()]
    column_a_data = list(data[column_a])
    column_b_data = list(data[column_b])
    bleu = moses_multi_bleu(column_a_data, column_b_data, True)
    print('BLEU: ', bleu)


if __name__ == '__main__':
    main()
