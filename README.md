# ProfNER-ST Baseline 1 - Lookup

## Introduction
This system extracts information from a set of annotated documents. Then, it checks whether, in a new set of text documents, the extracted annotation are present.
This system works as baseline for ProfNER shared task [http://temu.bsc.es/smm4h-spanish/]

#### Steps: 
1. Extract annotations from tab-separated file and tokenize them.
2. For a new file, tokenize words. 
3. Get the intersection between tokens in annotations and tokens in words.
4. For a token in the intersection, check surroundings of every occurrence in the text, to confirm whether there is a match with any annotation.
5. Repeat step 4 for every token in the intersection.
6. Repeat steps 2-5 for every file in the directory.

#### Input format
+ Gold standard: tab-separated file with the annotations. Format (contains headers):
```
tweet_id	begin	end	type	extraction
```

+ Text files where codes will be predicted.


#### Output format
+ Tab-separated file with annotations. There are two possible formats, depending on the sub-track we use. Format for sub-track a on tweet classification:
```
tweet_id	label
```
Format for sub-track b on Named Entity Recognition detection and classification:
```
tweet_id	begin	end	type	extraction
```

## Getting Started

Scripts written in Python 3.7, anaconda distribution Anaconda3-2019.07-Linux-x86_64.sh

### Prerequisites

You need to have installed python3 and its base libraries, plus:
+ pandas
+ os
+ time
+ re
+ string
+ unicodedata
+ spacy

### Installing

```
git clone <repo_url>
```

## Usage

Both scripts accept the same two parameters:
+ --gs_path (-gs) specifies the path to the Gold Standard file.
+ --gs_path2 (-gs2) specifies the path to an additional GS file (not mandatory parameter).
+ --data_path (-data) specifies the path to the text files.
+ --out_path (-out) specifies the path to the output predictions file.
+ --sub_track (-t) specifies the task we are using the system for. In ProfNER Track, there are 2 tasks (1 for tweet classification and 2 for NER).

```
$> python lookup.py -gs gold_standard.tsv -data datapath/ -out predictions.tsv -t TASK_NUMBER
```

## Contact
Antonio Miranda (antonio.miranda@bsc.es)
