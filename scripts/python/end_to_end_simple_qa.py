import pprint

from elasticsearch_dsl.connections import connections
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

from lib.checkpoint import Checkpoint

pretty_printer = pprint.PrettyPrinter(indent=2)

# Load models into memory
RELATION_CLASSIFIER = '../../results/0626.11-05_08:38:02.relation_classifier/11m_05d_08h_45m_57s.pt'
OBJECT_RECOGNITION = '../../results/0605.11-05_09:35:18.object_recognition/11m_05d_09h_45m_22s.pt'

relation_classifier_predict = Checkpoint(checkpoint_path=RELATION_CLASSIFIER).predict
object_recognition_predict = Checkpoint(checkpoint_path=OBJECT_RECOGNITION).predict


def preprocess(s):
    """
    This preprocessing step is required for `get_object`. The same step is computed when creating the data.
    
    Args:
        s (str) string to preprocess
    Returns:
        preprocessed string
    """
    s = s.lower()
    s = s.replace('?', '')
    s = s.replace('.', '')
    s = s.strip()
    return s


def get_object(question):
    """ Given a question return the object in the question using `OBJECT_CHECKPOINT` model. """
    question = preprocess(question)
    marks, confidence = object_recognition_predict(question)
    marks = marks.split()
    question = question.split()
    entity = []
    for marker, word in zip(marks, question):
        if marker == 'e':
            entity.append(word)
    return ' '.join(entity), sum(confidence) / len(confidence)


def get_relation(question, top_k=3):
    """ 
    Given a question return the predicate in the question using `RELATION_CLASSIFIER` model.
    
    Args:
        question (str)
    Returns:
        list of predicates and their confidence
    """
    question = preprocess(question)
    return [(class_[0], confidence[0])
            for class_, confidence in list(relation_classifier_predict(question, top=top_k))]


# Test models
print(get_relation('what area code is 845'))
print(get_object('Who is Obama?'))

# Define a default Elasticsearch client
connections.create_connection(hosts=['localhost'])
client = Elasticsearch()


def get_object_link(object_, property_, print_info=False):
    """
    Link the object_ (str) to a QID.
    
    Args:
        object_ (str): object to link
        predicate_mid (str): MID to filter the results
        print_info (bool): print the top matches
    Return:
        mid (str): object QID
        score (float): the score assigned to the top result
        name (str): name of the object QID
    """
    property_ = property_.replace('www.freebase.com/', '')
    print(object_, property_)
    hits = Search.from_dict({
        'query': {
            'bool': {
                'filter': {
                    'nested': {
                        'path': 'facts',
                        'query': {
                            'match': {
                                'facts.property': property_
                            }
                        }
                    }
                },
                'must': [{
                    'match': {
                        'name': object_
                    },
                }]
            }
        },
    }).using(client).index('fb5m_entities').execute()
    if print_info:
        print('Hits:')
        pretty_printer.pprint([(hit.mid, hit.name, hit.meta.score, hit.facts) for hit in hits])
    if len(hits) == 0:
        print('WARNING: No Object found.')
        return None, None, None, None
    return hits[0].mid, hits[0].meta.score, hits[0].name, hits[0].facts


# In[146]:

import ujson as json
import pandas as pd

from IPython.display import display

SRC = '/Users/petrochuk/sync/pytorch-seq2seq/data/simple_qa/dev.tsv'


def answer_question(question):
    """
    Answer a question in JSON format for IO.
    
    Args:
        question (str)
    Returns:
        DL Predicate (str): name of the predicate
        DL Preidcate PID (str): PID of the predicate
        DL Preidcate Confidence (float)
        DL Top Predicates (list of predicate, PID, and confidence): Top predicates about > .9 confience
        DL Object (str): name of the object in the question
        DL Object Confidence (float)
        DL Object Name (str): name of the Wikidata object linked too
        DL Object Aliases (list of str): list of aliases for the object
        DL Object ID (str): QID for th WikiData object
        DL Object Score (float): score by ElasticSearch for object linking
        DL Answers (list of Object ID, Score, Answer): list of tuples that have the PID and QID
    """
    print()
    question = preprocess(question)
    print('Question:', question)
    top_predicates = get_predicate_id(question, top=3)
    print('Predicate QIDs:', top_predicates)
    top_mid = None
    top_score = 0
    top_name = None
    top_predicate_id = None
    top_subject = None
    object_, object_confidence = get_object(question)
    print('Object:', object_, '| Confidence:', object_confidence)
    for predicate_id, confidence in top_predicates:
        mid, score, name, facts = get_object_link(object_, predicate_id)
        print('Object:', name)
        print('Object Score:', score)
        print('Predicate:', predicate_id)
        print('')
        if mid and score > top_score:
            top_subject = [fact for fact in facts if fact['property'] in predicate_id]
            top_mid = mid
            top_score = score
            top_name = name
            top_predicate_id = predicate_id
    print('Object MID:', top_mid)
    print('Object Score:', top_score)
    print('Object Name:', top_name)
    print('Top Predicate:', top_predicate_id)
    if top_subject:
        print('Top Subject:', top_subject[0]['subjects'])
    return top_mid, top_predicate_id


def main():
    """
    Run main to save answers to the pandas table SRC
    """
    data = pd.read_table(SRC)
    display(data.head())

    def add_answers(row):
        question = row['Question EN']
        mid, predicate_id = answer_question(question)
        object_mid = row['Object MID'].replace('www.freebase.com/m/', '').strip()
        if not mid or mid != object_mid or row['Freebase Property'] != predicate_id:
            print('FAIL!')
            if mid != object_mid:
                print('WRONG MID:', mid)
            if row['Freebase Property'] != predicate_id:
                print('WRONG PROP:', predicate_id)
            print('Correct MID:', object_mid)
            print('Correct Object:', row['Object EN'])
            print('Correct Subject:', row['Subject EN'])
            print('Correct Subject:', row['Subject MID'])
            print('Correct Property:', row['Freebase Property'])
        else:
            print('CORRECT!')

    data = data.apply(add_answers, axis=1)
    print('Done!')


main()
