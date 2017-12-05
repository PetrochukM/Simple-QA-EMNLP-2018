from tqdm import tqdm_notebook
from numpy import nan
from IPython.display import display, Markdown
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import pandas as pd
import numpy as np

stem = SnowballStemmer('english', ignore_stopwords=True).stem


def get_alias_in_sentence(sentence, aliases):
    """
    Get the longest alias in the sentence.

    To align the alias and the sentence:
        - ignore possessive tokens
        - lowercase strings
        - stem words
    """
    aliases = sorted(aliases, key=lambda k: len(k), reverse=True)  # Sort by longest
    possesives = ["'", "'s", "`s", "`"]
    token_sentence = [t for t in word_tokenize(sentence) if t not in possesives]
    for i, alias in enumerate(aliases):
        token_alias = [t for t in word_tokenize(alias) if t not in possesives]
        for i, token in enumerate(token_sentence):
            # No room for spacy alias
            if i + len(token_alias) > len(token_sentence):
                break

            # Check if spacy alias matches
            for j, other_token in enumerate(token_alias):
                offset_token = token_sentence[i + j]
                if stem(offset_token) != stem(other_token):
                    break
                elif j == len(token_alias) - 1:  # Last iteration
                    return alias
    return nan


def add_subject_name(df, cursor, print_=False):
    n_failed_no_subject_reference = 0
    n_failed_no_alias = 0
    subject_names = []
    print_data = []

    for index, row in tqdm_notebook(df.iterrows(), total=df.shape[0]):
        cursor.execute("SELECT alias FROM fb_name WHERE mid=%s", (row['subject'],))
        rows = cursor.fetchall()
        if len(rows) == 0:
            n_failed_no_alias += 1
            if print_:
                print('Subject MID (%s) does not have aliases.' % row['subject'])
            subject_names.append(nan)
            df.loc[index, 'subject_name'] = nan
            continue
        aliases = [row[0].strip() for row in rows]
        alias = get_alias_in_sentence(row['question'], aliases)
        if not isinstance(alias, str):
            n_failed_no_subject_reference += 1
            print_data.append({
                'Question': row['question'],
                'Subject': row['subject'],
                'Aliases': aliases,
            })
        subject_names.append(alias)

    df['subject_name'] = pd.Series(subject_names, index=df.index)

    if print_:
        display(pd.DataFrame(print_data)[:50])
        percent_failed_no_subject_reference = (n_failed_no_subject_reference / df.shape[0]) * 100
        percent_failed_no_alias = (n_failed_no_alias / df.shape[0]) * 100
        display(
            Markdown('### Numbers' + '\n%f%% [%d of %d] questions do not reference subject' %
                     (percent_failed_no_subject_reference, n_failed_no_subject_reference,
                      df.shape[0]) + '\n\n%f%% [%d of %d] subject mids do not have aliases' %
                     (percent_failed_no_alias, n_failed_no_alias, df.shape[0])))
