"""
translate.py makes repeated requests to DeepL to translate a set of data.
"""

import json
import requests
import pandas
import progressbar


def translate(batch):
    """ Translate a batch of questions from english to french using DeepL 44 BLEU """
    headers = {
        'cookie':
        'LMTBID="efd84772-1e6e-4a72-be20-a4478c3b8f0d"; selectedTargetLang=FR; preferredLangs=FR%2CEN',
        'origin':
        'https://www.deepl.com',
        'accept-encoding':
        'gzip, deflate, br',
        'accept-language':
        'en-US,en;q=0.9',
        'user-agent':
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36',
        'content-type':
        'text/plain',
        'accept':
        '*/*',
        'referer':
        'https://www.deepl.com/translator',
        'authority':
        'www.deepl.com',
        'x-requested-with':
        'XMLHttpRequest',
    }

    jobs = [{
        "kind": "default",
        "raw_en_sentence": sentence
    } for sentence in batch]
    data = {
        "jsonrpc": "2.0",
        "method": "LMT_handle_jobs",
        "params": {
            "jobs": jobs,
            "lang": {
                "user_preferred_langs": ["EN"],
                "source_lang_user_selected": "auto",
                "target_lang": "FR"
            },
            "priority": -1
        },
        "id": 13
    }
    result = requests.post(
        'https://www.deepl.com/jsonrpc',
        headers=headers,
        data=json.dumps(data)).json()
    translations = []
    for i, translation in enumerate(result['result']['translations']):
        if translation['beams']:
            translations.append(
                translation['beams'][0]['postprocessed_sentence'])
        else:
            # No translation is returned. Example:
            # which publisher was behind  age of conan: unchained
            translations.append('')
            print('[ERROR] NO TRANSLATION FOUND:')
            print('> ', translation)
            print('> ', batch[i])
    assert len(translations) == len(batch)
    return translations


def main(src='data/simple_qa/test.tsv',
         dest='data/simple_qa/test_temp.tsv',
         src_column='Question EN',
         dest_column='Question FR DeepL',
         character_batch_max=5000):
    """ Runs through src translating from src_column to dest_column then saving to dest """
    data = pandas.read_table(src)
    batch = []  # Batch to reduce the number of internet requests
    character_count = 0  # DeepL has a character limit
    translated = []
    _bar = progressbar.ProgressBar(
        redirect_stdout=True,
        max_value=len(data.index),
        widgets=[
            '[',
            progressbar.Timer(),
            '] ',
            progressbar.Bar(),
            ' (',
            progressbar.ETA(),
            ') ',
        ])

    # Batch translate
    for i, row in _bar(data.iterrows()):
        question = row[src_column]
        if character_count + len(question) >= character_batch_max:
            print('[TRANSLATING] %d characters' % character_count)
            translated.extend(translate(batch))
            batch = []
            character_count = 0
        batch.append(question)
        character_count += len(question)
    translated.extend(translate(batch)) # Last batch

    # Save data
    assert len(translated) == len(data.index)
    for i, translation in enumerate(translated):
        data.set_value(i, dest_column, translation)
    data.to_csv(dest, sep='\t', index=False)


if __name__ == '__main__':
    main()
