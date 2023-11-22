import os
import re
import json
import logging
import torch 

logger = logging.getLogger(__name__)

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def remove_articles(_text):
    return RE_ART.sub(' ', _text)


def white_space_fix(_text):
    return ' '.join(_text.split())


def remove_punc(_text):
    return RE_PUNC.sub(' ', _text)  # convert punctuation to spaces


def lower(_text):
    return _text.lower()


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace. """
    return white_space_fix(remove_articles(remove_punc(lower(text))))


def write_generation_preds(output_file, dialog_ids, responses, ground_truths):
    new_labels = []
    for i, response in enumerate(responses):
        new_labels.append(
            {"id": dialog_ids[i], "ground_truth_sparql": ground_truths[i], "predicted_sparql": response})

    if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as jsonfile:
        logger.info("Writing predictions to {}".format(output_file))
        json.dump(new_labels, jsonfile, indent=2)


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]

    return arrays


def truncate_sequences(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences
 
    while words_to_cut > len(sequences[0]):
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]

    sequences[0] = sequences[0][words_to_cut:]
    return sequences


def extract_first_sparql(text):
    # Use regular expressions to find the content between '### SPARQL:' and the next '###'
    # First try to find the content between '### SPARQL:' and the next '###'
    match = re.search(r'### SPARQL:\s*(.+?)\s*###', text, re.DOTALL)

    # If not found, try finding content from '### SPARQL:' till end of string
    if not match:
        match = re.search(r'### SPARQL:(.+)', text, re.DOTALL)

    # If not found, try finding content from '### Output:' till end of string
    if not match:
        match = re.search(r'### Output:\s*(.+?)\s*###', text, re.DOTALL)
    if not match:
        match = re.search(r'### Output:(.+)', text, re.DOTALL)

    # If not found, try finding content from '### Answer:' till end of string
    if not match:
        match = re.search(r'### Answer:\s*(.+?)\s*###', text, re.DOTALL)
    if not match:
        match = re.search(r'### Answer:(.+)', text, re.DOTALL)

    # If not found, try finding content from '### Answer:' till end of string
    if not match:
        match = re.search(r'### ANSWER:\s*(.+?)\s*###', text, re.DOTALL)
    if not match:
        match = re.search(r'### ANSWER:(.+)', text, re.DOTALL)

    # If not found, try finding content from '### SPARQL query:' till end of string
    if not match:
        match = re.search(r'### SPARQL query:\s*(.+?)\s*###', text, re.DOTALL)
    if not match:
        match = re.search(r'### SPARQL query:(.+)', text, re.DOTALL)

    # If a match is found, return it, otherwise return None
    if match:
        return match.group(1).strip()
    return None




def extract_first_sparql_bgee(text):
    # First, try to find the content that starts with 'PREFIX' until the end of the string or until '###'
    match = re.search(r'PREFIX\s*(.+?)(\s*###|$)', text, re.DOTALL)
    
    # If not found, then proceed with the other searches as before
    if not match:
        # Search between '### SPARQL:' and the next '###'
        match = re.search(r'### SPARQL:\s*(.+?)\s*###', text, re.DOTALL)
        if not match:
            # If not found, try finding content from '### SPARQL:' till end of string
            match = re.search(r'### SPARQL:(.+)', text, re.DOTALL)

        # If not found, try finding content from '### Output:' till next '###'
        if not match:
            match = re.search(r'### Output:\s*(.+?)\s*###', text, re.DOTALL)
        if not match:
            # If not found, try finding content from '### Output:' till end of string
            match = re.search(r'### Output:(.+)', text, re.DOTALL)

        # If not found, try finding content from '### Answer:' till next '###'
        if not match:
            match = re.search(r'### Answer:\s*(.+?)\s*###', text, re.DOTALL)
        if not match:
            # If not found, try finding content from '### Answer:' till end of string
            match = re.search(r'### Answer:(.+)', text, re.DOTALL)

        # If not found, try finding content from '### SPARQL query:' till next '###'
        if not match:
            match = re.search(r'### SPARQL query:\s*(.+?)\s*###', text, re.DOTALL)
        if not match:
            # If not found, try finding content from '### SPARQL query:' till end of string
            match = re.search(r'### SPARQL query:(.+)', text, re.DOTALL)

    # If a match is found, return it, otherwise return None
    if match:
        return match.group(1).strip()
    return None



def extract_first_sparql_contrastive(text):
    # Capture the content after 'Correct query:' and 'Incorrect query'
    # Capture the content after 'Correct query:'
    correct_query_match = re.search(
        r'Correct query:\s*(.+?)(?:\nIncorrect query|\Z)', text, re.DOTALL)

    # If a match for the correct query is found, return it
    if correct_query_match:
        return correct_query_match.group(1).strip()
    return None

def load_data_from_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Convert list of dicts to dict of lists
    return {key: [d[key] for d in data] for key in data[0].keys()}


def generate_prompt(example):
    question = example['question']
    sparql = example["sparql"]
    prompt = f"### INSTRUCTION\nPlease convert the following context into an SPARQL query.\n\n### CONTEXT:\n{question}\n\n### SPARQL:\n{sparql}"
    return {'text': prompt}

def prompt_question(question):
    return f"### INSTRUCTION\nPlease convert the following context into an SPARQL query.\n\n### CONTEXT:\n{question}\n\n### SPARQL:"




def prompt_question_contrastivev1(question):
    """### INSTRUCTION:
                Generate a correct SPARQL query that returns the answer of the following question. Generate four incorrect SPARQL queries of different types.
                
                ### Input:
                {question}
                
                ### SPARQL:
                {sparql}

                ### INCORRECT SPARQL
                Incorrect query 1: {inc_sparqls[0]}
                Incorrect query 2: {inc_sparqls[1]}
                Incorrect query 3: {inc_sparqls[2]}
                Incorrect query 4: {inc_sparqls[3]}
                """

    return f"### INSTRUCTION\nGenerate a correct SPARQL query that returns the answer of the following question. Generate four incorrect SPARQL queries of different types.\n\n### Input:\n{question}\n\n### SPARQL:\n"

import requests
def query_plain(text, url="http://localhost:8888/plain"):
    return requests.post(url, json={'text': text}).json()

def prompt_append_BERN2EL(question,type='plain'):
    """
    Uses BERN2 to generate entity information to the query
    https://github.com/dmis-lab/bern2
    It needs the server up with be run_bern2.sh
    """
    annotations = query_plain(question)['annotations']
    if type == 'plain':
        entities_info = ''
        for mention in annotations:
            entities_info += 'The entity "' + mention['mention'] + '" is of type "' +\
                mention['obj'] + '" with id "' + mention['id'][0] + '"\n'
    elif type == 'dict':
        entities_info = annotations

    return f"""
    ### INSTRUCTION:
    Generate a correct SPARQL query that returns the answer of the following question. Generate four incorrect SPARQL queries of different types.
    
    ### Input:
    {question}
    
    ### Entity information of Input
    
    {entities_info}
    """
#
def prompt_append_SpacyWikidataEL(question,type='plain',nlp=None):
    """
    wikidata entity linking
    """
    doc = nlp(question)
    entities_info = ''
    for span in doc.ents:
        description = span._.description[0] if span._.description else "No description available"
        entities_info += 'The entity "' + span.text + '" is of type "' +\
            span.label_ + '" with id "' + span.kb_id_ + \
            '" with description "' + description + '"\n'
        # print((span.text, span.kb_id_, span.label_,
        #     span._.description[0], span._.score))

    return f"""
    ### INSTRUCTION:
    Generate a correct SPARQL query that returns the answer of the following question. Generate four incorrect SPARQL queries of different types.
    
    ### Input:
    {question}
    
    ### Entity information of Input
    
    {entities_info}
    """



def truncate_batch(batch, max_length=None):
    """
    To remove tokens  before the padding '32000' which cause
    Assertion `srcIndex < srcSelectDimSize` failed.
    """
    lengths = batch['attention_mask'].sum(dim=1)

    # If max_length is not provided, take the minimum of the lengths in the batch
    if not max_length:
        max_length = lengths.min().item()

    # Slice the tensors
    batch['input_ids'] = batch['input_ids'][:, :max_length]
    batch['attention_mask'] = batch['attention_mask'][:, :max_length]
    return batch


