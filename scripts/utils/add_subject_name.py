import pandas as pd
from tqdm import tqdm_notebook
from numpy import nan
from IPython.display import display, Markdown
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()
stemmer = SnowballStemmer('english', ignore_stopwords=True)


def stem_phrase(s):
    s = s.lower()
    if len(s) >= 1 and s[-1] in ['?', '.', '!']:  # Remove end of sentence punctuation
        s = s[0:-1]
    # https://stackoverflow.com/questions/34714162/preventing-splitting-at-apostrophies-when-tokenizing-words-using-nltk
    s = [word.strip() for word in tokenizer.tokenize(s)]
    s = [stemmer.stem(word) for word in s]
    return ' '.join(s)


def get_name_in_question(question, names):
    for i, name in enumerate(names):
        if name in question:
            return i
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
        names = [row[0].strip() for row in rows]
        names = sorted([name for name in names], key=lambda k: len(k), reverse=True)
        stemmed_names = [stem_phrase(name) for name in names]
        stemmed_question = stem_phrase(row['question'])
        name_index = get_name_in_question(stemmed_question, stemmed_names)
        if name_index is nan:
            n_failed_no_subject_reference += 1
            print_data.append({
                'Question Stemmed': stemmed_question,
                'Question': row['question'],
                'Subject': row['subject'],
                'Names Stemmed': stemmed_names,
                'Names': names,
            })
            subject_names.append(nan)
        else:
            subject_names.append(names[name_index])

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
