# -*- coding:utf8 -*-
import codecs
import json
import sys
import nltk
import re
from tqdm import tqdm   
from rouge import Rouge
import numpy as np
import random


def read_metadata(file_to_read):
    passage_list     =  []
    answers_list     =  []
    query_list       =  []
    query_id_list    =  []
    
    with open(file_to_read, 'r', encoding='utf8') as data_file:
        for line in data_file:

            instances = json.loads(line)['data']
            for i, instance in enumerate(instances):
                for para in instance['paragraphs']:
                    for i,qas in enumerate(para['qas']):
                        passage = para['context'].replace("''", '"').replace("``", '"')
                        passage_list.append(passage)
                        answers_list.append(qas['answers'][0]['text'])
                        query_list.append(qas['question'])
                        query_id_list.append(qas['id'])
    data_dict = {}
    data_dict['passages']  = passage_list
    data_dict['answers']   = answers_list
    data_dict['questions'] = query_list
    data_dict['q_id']      = query_id_list
    return data_dict  


def renew_data_dict(train_data_dict):
	indics, bad_indics = get_indics(train_data_dict['passages'],train_data_dict['answers'])
	renew_passage_list = []
	renew_question_list = []
	for i, passage in enumerate(train_data_dict['passages']):
		if i not in bad_indics:
			renew_passage_list.append(passage)
			renew_question_list.append(train_data_dict['questions'][i])

	train_data_dict['ans_indics'] = indics
	train_data_dict['passages'] = renew_passage_list
	train_data_dict['questions'] = renew_question_list  
	return train_data_dict      
if __name__ == '__main__':
	path = '''/home/zhangs/RC/SQUAD_data/train-v1.1.json'''
	train_data_dict = read_metadata(path)
	train_data_dict = renew_data_dict(train_data_dict)
	print(len(train_data_dict['ans_indics']))
	print(len(train_data_dict['questions']))


## signals are also regarded "words"
def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
def Tokenize(para_list):
    
    if isinstance(para_list, list) and isinstance(para_list[0], str):
        l = []
        # split a paragraph by sentences
        sent_tokenize = nltk.sent_tokenize

        for string in para_list:
            li_item = []
            for item in list(map(word_tokenize, sent_tokenize(string))):
                li_item.append(process_tokens(item))
            l.append(li_item)
        return l
    elif isinstance(para_list, str):
        sent_tokenize = nltk.sent_tokenize
        li_item = []
        for item in list(map(word_tokenize, sent_tokenize(para_list))):
            li_item.append(process_tokens(item))
        return li_item
    else:
        raise Exception
def Tokenize_without_sent(para_list):
    if isinstance(para_list, list) and isinstance(para_list[0], str):
        l = []
        for string in para_list:
            l.append(Tokenize_string_word_level(string))
        return l
    elif isinstance(para_list, str):
        return Tokenize_string_word_level(para_list)
    else:
        raise Exception
def Tokenize_string_word_level(para):
    l = process_tokens(word_tokenize(para))
    return l
'''
this method is used to split '/' or '-', 
eg: It's 2017/09/06  or 1997-2017
'''
def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0", "\*")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        
        # tokens.extend(re.split("([{}])".format("".join(l)), token))
        temp = re.split("([{}])".format("".join(l)), token)
        for item in temp:
            if len(item) > 1 and item[len(item)-1] == '.':
                tokens.append(item[:len(item)-1])
                tokens.append('.')
            elif len(item) > 0:
                tokens.append(item)
    return tokens
def get_indics(para_list, ans_list):
    indics_list = []
    fucking_indics = []
    for i, para in enumerate(para_list):
        tokenized_para = Tokenize_string_word_level(para)
        tokenized_ans = Tokenize_string_word_level(ans_list[i])
        index_start, index_stop = get_idx_sublist(tokenized_para, tokenized_ans)
        if index_start == -1:
            fucking_indics.append(i)
        else:   
            indics_list.append([index_start, index_stop])
    return indics_list, fucking_indics
## the case of words should be taken into consideration
def get_word2idx_and_embmat(path_to_file, path_vocab):
    vocab_list = []
    with open(path_vocab, 'r') as vocab_file:
        for word in vocab_file:
            vocab_list.append(word[:word.index('\n')])        
    
    word2idx_dict = {}
    for i, word in enumerate(vocab_list):
        word2idx_dict[word] = i

    word2vec_dict = {}
    with open(path_to_file, 'r') as vec_file:
        for line in tqdm(vec_file):
            list_of_line = line.split(' ')
            word2vec_dict[list_of_line[0]] = list(map(float, list_of_line[1:]))
    emb_mat = []
    for i, word in enumerate(vocab_list):
        if word not in word2vec_dict:
            emb_mat.append([0 for i in range(100)])
        else:
            emb_mat.append(word2vec_dict[word])

    emb_mat = np.asarray(emb_mat)
    emb_mat = emb_mat.astype(dtype='float32')
    
    return word2idx_dict, emb_mat

def get_char2idx(data_dict):
    char2idx_dict = {}
    i = 0
    for key in data_dict:
        if key == 'passages' or key == 'questions':
            for string in data_dict[key]:
                for char in string:
                    if char not in char2idx_dict:
                        char2idx_dict[char] = i
                        i+=1
    char_vocabulary_size = len(char2idx_dict)
    return char2idx_dict, char_vocabulary_size


def write_to_file( path, data):
        with open(path, 'w', encoding='utf8') as data_file:
            data_file.write(json.dumps(data))  

def get_random_eles_from_list(list_to_select, num_ele):
    return random.sample(list_to_select, num_ele)



def get_flat_idx(wordss, idx):
    return sum(len(words) for words in wordss[:idx[0]]) + idx[1]
def get_phrase(context, words, span):

    #span looks like: [ start_index, stop_index ]
    flat_start, flat_stop = span
    #get 1d index in the passage

    if flat_start > flat_stop:
        k = flat_start
        flat_start = flat_stop
        flat_stop = k
    
    flat_stop += 1
    char_idx = 0
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        char_idx = context.find(word, char_idx)
        assert char_idx >= 0
        if word_idx == flat_start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == flat_stop - 1:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]

def get_y_index(y_after_softmax):
    y_indics = []
    for y in y_after_softmax:
        max_value = 0
        for j, word in enumerate(y):
            if word > max_value:
                max_value = word
                word_index = j
        # print(max_value)
        y_indics.append(word_index)
    return y_indics

def get_idx_sublist(li, subli):
    for idx_li in range(len(li)):
        flag = 1
        if idx_li+len(subli) > len(li):
            return -1, -1
        for idx_subli in range(len(subli)):

            if subli[idx_subli] != li[idx_li+idx_subli]:
                flag = 0
                break

        if flag == 1:
            return idx_li, idx_li+len(subli)-1
        else:
            continue
    return -1, -1  