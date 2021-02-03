#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:10:50 2020

@author: antonio
"""
import unicodedata
import string
from spacy.lang.es import STOP_WORDS
import re


def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.printable)

def Flatten(ul):
    '''
    DESCRIPTION: receives a nested list and returns it flattened
    INPUT: ul: list
    OUTPUT: fl: list'''
    
    fl = []
    for i in ul:
        if type(i) is list:
            fl += Flatten(i)
        else:
            fl += [i]
    return fl


def format_ann_info(df_annot, min_upper):
    '''
    DESCRIPTION: Build useful Python dicts from DataFrame with info from TSV file
    
    INPUT: df_annot: pandas DataFrame with 4 columns: 'filename', 'label', 'code', 'span'
           min_upper: int. Specifies the minimum number of characters of a word
               to lowercase it (to prevent mistakes with acronyms).
    
    OUTPUT: annot2label: python dict with every unmodified annotation and 
              its label.
            annot2annot_processed: python dict with every unmodified annotation
              and the words it has normalized.
            annotations_final: set of word in annotations.
    '''
    df_annot.columns = ['filename', 'pos0', 'pos1', 'label', 'span']
    df_annot = df_annot.drop(['pos0', 'pos1'], axis=1)
 
    set_annotations = set(df_annot.span)
    
    annot2label = dict(zip(df_annot.drop_duplicates(subset=['span', 'label']).span,
                           df_annot.drop_duplicates(subset=['span', 'label']).label))
    
    annot2annot = dict(zip(set_annotations, set_annotations))
    
    # Split values: {'one': 'three two'} must be {'one': ['three', 'two']}   
    annot2annot_split = annot2annot.copy()
    annot2annot_split = dict((k, v.split()) for k,v in annot2annot_split.items())
    
    # Do not store stopwords or single-character words as values
    for k, v in annot2annot_split.items():
        annot2annot_split[k] = list(filter(lambda x: x not in STOP_WORDS, v))
    for k, v in annot2annot_split.items():
        annot2annot_split[k] = list(filter(lambda x: len(x) > 1, v))
    
    # Trim punctuation or multiple spaces
    annot2annot_trim = annot2annot.copy()
    for k, v in annot2annot_split.items():
        annot2annot_trim[k] = list(map(lambda x: x.strip(string.punctuation + ' '), v))
        
    # Lower case values
    annot2annot_lower = annot2annot_trim.copy()
    for k, v in annot2annot_trim.items():
        annot2annot_lower[k] = list(map(lambda x: x.lower() if len(x) > min_upper else x, v))
    
    # remove accents from annotations
    annot2annot_processed = annot2annot_lower.copy()
    for k, v in annot2annot_lower.items():
        annot2annot_processed[k] = list(map(lambda x: remove_accents(x), v))
        
    # Get list of all processed words in annotations (except STOPWORDS or single-character)
    annotations_final = set(Flatten(list(annot2annot_processed.values())))
    
    return annot2label, annot2annot_processed, annotations_final


def format_text_info(txt, min_upper):
    '''
    DESCRIPTION: 
    1. Obtain list of words of interest in text (no STPW and longer than 1 character)
    2. Obtain dictionary with words of interest and their position in the 
    original text. Words of interest are normalized: lowercased and removed 
    accents.
    
    INPUT: txt: str with the text to format.
           min_upper: int. Specifies the minimum number of characters of a word
               to lowercase it (to prevent mistakes with acronyms).
    
    OUTPUT: words_processed2pos: dictionary relating the word normalzied (trimmed,
                removed stpw, lowercased, removed accents) and its position in
                the original text.
            words_final: set of words in text.
    '''
    
    # Get individual words and their position in original txt
    words = txt.split()
    
    # Remove beginning and end punctuation and whitespaces. 
    words_no_punctuation = list(map(lambda x: x.strip(string.punctuation + ' '), words))
    
    # Remove stopwords and single-character words
    large_words = list(filter(lambda x: len(x) > 1, words_no_punctuation))
    words_no_stw = set(filter(lambda x: x.lower() not in STOP_WORDS, large_words))
    
    # Create dict with words and their positions in text
    words2pos = {}
    for word in words_no_stw:
        occurrences = list(re.finditer(re.escape(word), txt))
        if len(occurrences) == 0:
            print(word)
            raise ValueError('ERROR: ORIGINAL WORD NOT FOUND IN ORIGINAL TEXT')
        pos = list(map(lambda x: x.span(), occurrences))
        words2pos[word] = pos

    # lowercase words and remove accents from words
    words_processed2pos = dict((remove_accents(k.lower()), v) if len(k) > min_upper else 
                                (k,v) for k,v in words2pos.items())
    
    # Set of transformed words
    words_final = set(words_processed2pos)
    
    return words_final, words_processed2pos



def adjacent_combs(text, tokens2pos, n_words):
    '''
    DESCRIPTION: obtain all token combinations in a text. The maximum number
    of tokens in a combination is given by n_words.
    For example: text = 'buenos días míster jones', n_words = 3.
    output: [buenos, buenos días, buenos días míster, días, días míster, 
    días míster jones, míster, míster jones, jones]
    
    INPUT: text: string with full text
           tokens2pos: dictionary relating every token with its position in 
                  text. {tokens: (start, end)}
           n_words: maximum number of tokens in a combination
    
    OUTPUT: id2token_span: dictionary relating every token combination with an ID.
            id2token_span_pos: dictionary relating every token combination
                  (identified by an ID) with its position in the text.
            token_spans: list of token combinations.'''
    
    tokens = []
    for m in re.finditer(r'\S+', text):
        if all([i in string.punctuation for i in m.group()])==False:
            tokens.append(m.group())

    tokens_trim = list(map(lambda x: x.strip(string.punctuation), tokens))
    id2token_span = {}
    id2token_span_pos = {}
    count = 0
    
    tokens = tokens_trim
    
    for a in range(0, len(tokens)+1):
        for b in range(a+1, min(a + 1 + n_words, len(tokens)+1)):
            count = count + 1
            
            # Obtain combinations
            tokens_group = tokens[a:b] # Extract token group
            tokens_group = list(filter(None, tokens_group)) # remove empty elements
            
            if tokens_group:
                
                # Extract previous token
                token_prev = '' 
                if a>0:
                    c = 1
                    token_prev = tokens_trim[a-c:a][0]
                    # If token_prev is an empty space, it may be because there
                    # where a double empty space in the original text
                    while (token_prev == '') & (a>1):
                        c = c+1
                        token_prev = tokens_trim[a-c:a][0]
                        
                id2token_span[count] = tokens_group
                
                # Extract start and end positions
                if len(tokens_group) == 1:
                    pos = list(filter(lambda x: x[2] == token_prev, tokens2pos[tokens_group[0]]))
                    beg_pos = pos[0][0]
                    end_pos = pos[0][1]
                else:
                    beg =  list(filter(lambda x: x[2] == token_prev, tokens2pos[tokens_group[0]]))
                    end = list(filter(lambda x: x[2] == tokens_group[-2], tokens2pos[tokens_group[-1]]))
                    beg_pos = beg[0][0]
                    end_pos = end[0][1]
                    
                id2token_span_pos[count] = (beg_pos, end_pos) 

    token_spans = list(map(lambda x: ' '.join(x), id2token_span.values()))
    
    return id2token_span, id2token_span_pos, token_spans


def strip_punct(m_end, m_start, m_group, exit_bool):
    '''
    DESCRIPTION: remove recursively final and initial punctuation from 
              string and update start and end offset.
    
    INPUT: exit_bool: boolean value to tell whether to continue with the 
                      recursivety.
          m_end: end offset
          m_start: start offset
          m_group: string
    
    OUTPUT: exit_bool: boolean value to tell whether to continue with the 
                      recursivety.
          m_end: end offset
          m_start: start offset
          m_group: string
    '''
    
    if m_group[-1] in string.punctuation:
        m_end = m_end - 1
        m_group = m_group[0:-1]
        m_start = m_start
        exit_bool = 0
    elif m_group[0] in string.punctuation:
        m_end = m_end
        m_group = m_group[1:]
        m_start = m_start + 1
        exit_bool = 0
    else: 
        m_end = m_end
        m_group = m_group
        m_start = m_start
        exit_bool = 1
    if exit_bool == 0:
        m_end, m_start, m_group, exit_bool = strip_punct(m_end, m_start, m_group, exit_bool)
    return m_end, m_start, m_group, exit_bool


def tokenize_span(text, n_words):
    '''
    # DESCRIPTION: obtain all token combinations in a text and information 
    # about the position of every token combination in the original text.
    
    # INPUT: text: string
    #        n_words: int with the maximum number of tokens I want in a token
    #               combination.
    
    # OUTPUT: token_span2id: dictionary relating every token combination with an ID.
    #         id2token_span_pos: dictionary relating every token combination
    #               (identified by an ID) with its position in the text.
    #         token_spans: list of token combinations
    '''
    
    tokens2pos = {}
    
    # Split text into tokens (words), obtain their position and the previous token.
    m_last = ''
    for m in re.finditer(r'\S+', text):
        if all([i in string.punctuation for i in m.group()])==False:
            exit_bool = 0
            
            # remove recursively final and initial punctuation
            m_end, m_start, m_group, exit_bool = strip_punct(m.end(), m.start(), 
                                                             m.group(), exit_bool)
                
            # fill dictionary
            if m_group in tokens2pos.keys():
                tokens2pos[m_group].append([m_start, m_end, m_last])
            else:
                tokens2pos[m_group] = [[m_start, m_end, m_last]]
            m_last = m_group
        
    # Obtain token combinations
    id2token_span, id2token_span_pos, token_spans = adjacent_combs(text, 
                                                                   tokens2pos,
                                                                   n_words)
    
    # Reverse dict (no problem, keys and values are unique)
    token_span2id = {' '.join(v): k for k, v in id2token_span.items()}
    
    return token_span2id, id2token_span_pos, token_spans

def normalize_tokens(token_spans, min_upper):

    # DESCRIPTION: 
    token_span2token_span = dict(zip(token_spans, token_spans))
    
    
    # Lowercase
    token_span_lower2token_span = dict((k.lower(), v) if len(k) > min_upper else 
                                       (k,v) for k,v in token_span2token_span.items())

    # Remove whitespaces
    token_span_bs2token_span = dict((re.sub('\s+', ' ', k).strip(), v) for k,v 
                                    in token_span_lower2token_span.items())

    # Remove punctuation
    token_span_punc2token_span = dict((k.translate(str.maketrans('', '', string.punctuation)), v) for k,v in token_span_bs2token_span.items())
    
    # Remove accents
    token_span_processed2token_span = dict((remove_accents(k), v) for k,v in token_span_punc2token_span.items())
    
    return token_span_processed2token_span

def normalize_str(annot, min_upper):
    '''
    DESCRIPTION: normalize annotation: lowercase, remove extra whitespaces, 
    remove punctuation and remove accents.
    
    INPUT: annot: string
           min_upper: int. Specifies the minimum number of characters of a word
               to lowercase it (to prevent mistakes with acronyms).

    OUTPUT: annot_processed: string
    '''
    
    # Lowercase
    annot_lower = ' '.join(list(map(lambda x: x.lower() if len(x)>min_upper else x, annot.split(' '))))
    
    # Remove whitespaces
    annot_bs = re.sub('\s+', ' ', annot_lower).strip()

    # Remove punctuation
    annot_punct = annot_bs.translate(str.maketrans('', '', string.punctuation))
    
    # Remove accents
    annot_processed = remove_accents(annot_punct)
    
    return annot_processed


def eliminate_contained_annots(pos_matrix, new_annotations, off0, off1):
    '''
    DESCRIPTION: function to be used when a new annotation is found. 
              It check whether this new annotation contains in it an already 
              discovered annotation. In that case, the old annotation is 
              redundant, since the new one contains it. Then, the function
              removes the old annotation.
    '''
    to_eliminate = [pos for item, pos in zip(pos_matrix, range(0, len(new_annotations))) if (off0<=item[0]) & (item[1]<=off1)]
    new_annotations = [item for item, pos in zip(new_annotations, range(0, len(new_annotations))) if pos not in to_eliminate]
    pos_matrix = [item for item in pos_matrix if not (off0<=item[0]) & (item[1]<=off1)]
    
    return pos_matrix, new_annotations