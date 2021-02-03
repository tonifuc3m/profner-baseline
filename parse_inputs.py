#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:23:12 2020

@author: antonio
"""

import pandas as pd

def parse_tsv(tsv_path, sub_track):
    '''
    Parse TSV with annotations

    Parameters
    ----------
    tsv_path : list
        Path to annotations File
    sub_track: int
        Subtrack number (0 for tweet classification, 1 for NER)

    Returns
    -------
    df_annot : Pandas DataFrame
        Parse annotations

    '''
    df_annot = pd.read_csv(tsv_path, sep='\t', header=0)
    '''
    if sub_track == 2:
        df_annot = pd.read_csv(gs_path, sep='\t', header=0)
    elif sub_track == 1: 
        df_annot = pd.read_csv(gs_path, sep='\t', header=0) 
    else:
        raise ValueError('Incorrect sub-track value')
    '''
    
    return df_annot