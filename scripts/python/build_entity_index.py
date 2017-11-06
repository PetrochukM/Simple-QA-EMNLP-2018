""" Entity ElasticSearch Index

Here we build up an ElasticSearch cluster for linking to the FB5M knowledge base."""

from collections import defaultdict

import random
import pprint

from elasticsearch import Elasticsearch
from elasticsearch_dsl import DocType
from elasticsearch_dsl import Nested
from elasticsearch_dsl import Search
from elasticsearch_dsl import String
from elasticsearch_dsl.connections import connections
from elasticsearch.helpers import streaming_bulk

from tqdm import tqdm

FB5M = 'data/simple_qa/freebase-FB5M.txt'

# DOWNLOADED FROM: https://www.dropbox.com/s/yqbesl07hsw297w/FB5M.name.txt
MID_TO_NAME = 'data/simple_qa/FB5M.name.txt'

# Below we build an in memory representation of FB5M from the objects perspective.

object_to_fact = defaultdict(list)
for line in tqdm(open(FB5M, 'r'), total=12010500):
    split = line.split('\t')
    assert len(split) == 3, 'Malformed row'
    object_ = split[0].replace('www.freebase.com/m/', '').strip()
    property_ = split[1].replace('www.freebase.com/', '').strip()
    subjects = [url.replace('www.freebase.com/m/', '').strip() for url in split[2].split()]
    object_to_fact[object_].append({'property': property_, 'subjects': subjects})

pp = pprint.PrettyPrinter(indent=2)
print('Number of Objects:', len(object_to_fact))
print('Sample:', pp.pformat(random.sample(object_to_fact.items(), 5)))

# Below build a map to translate from MID to name. For each MID, we have multiple aliases that'll be
# used for entity linking.
mid_to_name = defaultdict(list)
for line in tqdm(open(MID_TO_NAME), total=5507279):
    split = line.strip().split('\t')
    mid = split[0].replace('<fb:m.', '').replace('>', '')
    name = split[2].replace('"', '').replace("'", '')
    mid_to_name[mid].append(name)
print('Number of entries:', len(mid_to_name))
print('Sample:', pp.pformat(random.sample(mid_to_name.items(), 10)))

# Lets check if our elasticsearch cluster is healthy; afterwards, we'll attempt to populate it with
# data.
# Define a default Elasticsearch client
connections.create_connection(hosts=['localhost'])

# Display cluster health
print('Health: %s', connections.get_connection().cluster.health())

client = Elasticsearch()
ENTITY_INDEX = 'fb5m_entities'

# check if last entity exists, then do not refetch
num_entities = 0
if client.indices.exists(index=ENTITY_INDEX):
    search = Search(using=client, index=ENTITY_INDEX)
    query = search.query("match_all")
    num_entities = query.count()
    print('Found %d documents in index "%s"' % (query.count(), ENTITY_INDEX))
else:
    print('%s index does not exist' % ENTITY_INDEX)


class FreebaseEntity(DocType):
    mid = String(index='not_analyzed')
    names = Nested(required=True, properties={'name': String()})
    facts = Nested(
        required=True,
        properties={
            'subjects': String(index='not_analyzed'),
            'property': String(index='not_analyzed')
        })

    class Meta:
        index = ENTITY_INDEX

    def save(self, **kwargs):
        return super().save(**kwargs)


# We checked if fb_5m exists, we checked that the cluster is healthy, and we setup a FreebaseEntity
# document schema. The last step in this process is to populate the cluster.


# save entities to elastic search
def get_entities():
    for mid, names in mid_to_name.items():
        if mid in object_to_fact:
            yield {'mid': mid, 'names': names, 'facts': object_to_fact[mid]}


#         else:
#             print('Lost MID:', mid, names)


def serialize_entity(mid, names, facts):
    """ serialize the instance into a dictionary so that it can be saved in elasticsearch. """
    names = [{'name': name} for name in names]
    return FreebaseEntity(mid=mid, facts=facts, names=names, meta={'id': mid}).to_dict(True)


def save_entities():
    """ efficiently save entities in bulk using `streaming_bulk` and `serialize_entity` """
    elasticsearch_connection = connections.get_connection()
    task_generator = (serialize_entity(**kwargs) for kwargs in get_entities())
    steaming_iterator = tqdm(
        streaming_bulk(
            elasticsearch_connection, task_generator, chunk_size=100, request_timeout=120))
    for ok, item in steaming_iterator:
        if not ok:
            print(item)


# save entities if not already saved
def create_index():
    input_ = input('WARNING - Delete %d entities? [YES/no] ' % num_entities)
    if input_ == 'YES':
        client.indices.delete(index=ENTITY_INDEX, ignore=[400, 404])
        # create the mappings in elasticsearch
        FreebaseEntity.init()
        save_entities()
    else:
        print('Not Deleting Index! Wohoo!')
    print('Done!')


create_index()
