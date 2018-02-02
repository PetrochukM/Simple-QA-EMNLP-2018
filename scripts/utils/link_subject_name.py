from tqdm import tqdm_notebook
from numpy import nan
from IPython.display import display, Markdown
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import pandas as pd

stem = SnowballStemmer('english', ignore_stopwords=True).stem
possesives = ["'", "'s", "`s", "`"]


def normalize_alias(s):
    """
    Following the same procedure for linking the subject name and question, we provide a method
    to normalize an alias for linking.

    To normalize alias we:
        - ignore possessive tokens
        - lowercase the string
        - stem words

    Returns: normalized string
    """
    if not isinstance(s, list):
        s = tokenize(str(s))

    words = [t.strip().lower() for t in s]
    words = [t for t in words if t not in possesives]
    words = [stem(t) for t in words]
    return ' '.join(words)


def tokenize(s):
    return word_tokenize(s)


def get_alias_in_sentence(sentence, aliases):
    """
    Get the longest alias in the sentence.

    To align the alias and the sentence:
        - ignore possessive tokens
        - lowercase the string
        - stem words
    """
    aliases = sorted(
        list(enumerate(aliases)), key=lambda k: len(k[1]), reverse=True)  # Sort by longest
    token_sentence = tokenize(sentence.lower())
    # Save the original index but get rid of the possesives
    token_sentence_no_poss = [(i, stem(t)) for i, t in enumerate(token_sentence)
                              if t not in possesives]

    for alias_index, alias in aliases:
        normalized_alias = normalize_alias(alias)
        token_alias = normalized_alias.split()

        for i, (original_i, token) in enumerate(token_sentence_no_poss):

            if i + len(token_alias) > len(token_sentence_no_poss):
                break  # No more room for alias in question

            # Check if spacy alias matches
            for j, other_token in enumerate(token_alias):
                offset_original_i, offset_token = token_sentence_no_poss[i + j]
                if offset_token != other_token:
                    break
                if j == len(token_alias) - 1:  # Last iteration
                    stop_index = offset_original_i + 1
                    start_index = original_i

                    # Here we ensure that given start_index and stop_index we can lookup the
                    # normalized alias
                    assert normalized_alias == normalize_alias(
                        token_sentence[start_index:stop_index]), """Normalized linking failed."""

                    return alias_index, start_index, stop_index
    return nan, nan, nan


def add_subject_name(df, cursor, print_=False, table='fb_name'):
    """
    Queries for potential aliases and links to the longest one referenced in the question.

    To link the alias and the sentence:
        - ignore possessive tokens
        - lowercase the string
        - stem words
    """
    n_failed_no_subject_reference = 0
    n_failed_no_alias = 0
    subject_names = []
    subject_names_start_index = []
    subject_names_stop_index = []
    print_data = []

    for index, row in tqdm_notebook(df.iterrows(), total=df.shape[0]):
        cursor.execute("SELECT alias FROM " + table + " WHERE mid=%s", (row['subject'],))
        rows = cursor.fetchall()

        # No aliases found
        if len(rows) == 0:
            n_failed_no_alias += 1
            subject_names.append(nan)
            subject_names_start_index.append(nan)
            subject_names_stop_index.append(nan)

            if print_:
                print('Subject MID (%s) does not have aliases.' % row['subject'])
        else:
            aliases = [row[0].strip() for row in rows]
            alias_index, start_index, stop_index = get_alias_in_sentence(row['question'], aliases)
            if not isinstance(alias_index, int):
                alias = nan
                n_failed_no_subject_reference += 1
                print_data.append({
                    'Question': row['question'],
                    'Subject': row['subject'],
                    'Aliases': aliases,
                })
            else:
                alias = aliases[alias_index]
                assert normalize_alias(alias) == normalize_alias(
                    tokenize(row['question'].lower())[start_index:stop_index])
            subject_names.append(alias)
            subject_names_start_index.append(start_index)
            subject_names_stop_index.append(stop_index)

    df['subject_name'] = pd.Series(subject_names, index=df.index)
    df['subject_name_start_index'] = pd.Series(subject_names_start_index, index=df.index)
    df['subject_name_stop_index'] = pd.Series(subject_names_stop_index, index=df.index)

    if print_:
        display(pd.DataFrame(print_data)[:50])
        percent_failed_no_subject_reference = (n_failed_no_subject_reference / df.shape[0]) * 100
        percent_failed_no_alias = (n_failed_no_alias / df.shape[0]) * 100
        display(
            Markdown('### Numbers' + '\n%f%% [%d of %d] questions do not reference subject' %
                     (percent_failed_no_subject_reference, n_failed_no_subject_reference,
                      df.shape[0]) + '\n\n%f%% [%d of %d] subject mids do not have aliases' %
                     (percent_failed_no_alias, n_failed_no_alias, df.shape[0])))
