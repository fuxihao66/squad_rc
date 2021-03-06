from rouge import Rouge
import re
from metadata_operation import *

def get_signal_idxs(string):
    pattern = re.compile(r'''[ ,()/~\u00B0\u2212\u2014\u2013\u201C\u2019\u201D\u2018-]''')
    signal_idx = 0
    list_of_signal_idxs = []
    while signal_idx < len(string):
        m = pattern.search(string[signal_idx+1:])
        if m:
            temp_idx = m.start()
            signal_idx = signal_idx+temp_idx+1
            
            if string[signal_idx] == ',' and signal_idx < len(string)-1:
                if string[signal_idx+1] == ' ':
                    list_of_signal_idxs.append(signal_idx)
            else:
                list_of_signal_idxs.append(signal_idx)
        else:
            break
    return list_of_signal_idxs
def get_rougel_score(summary, reference, score_type):
    rouge = Rouge()
    scores = rouge.get_scores(reference, summary)
    return scores[0]['rouge-l'][score_type]
def get_rougel_score_ave(summaries, references, score_type):
    rouge = Rouge()
    scores = rouge.get_scores(references, summaries, avg=True)
    return scores['rouge-l'][score_type]
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

def trans_idx_1dto2d(idx_start, idx_stop, list2d):
    max_num_sents = 22
    max_sent_size = 100

    flag = -1
    for i, ele in enumerate(list2d):
        for j, item in enumerate(ele):
            flag += 1

            if flag == idx_start:
                start_idxs_2d = [i, j]
                
            if flag == idx_stop:
                end_idxs_2d = [i, j]
    # print(flag)
    # print(idx_start)
    # print(idx_stop)

    return [start_idxs_2d, end_idxs_2d]
def get_highest_rl_span(para, reference, max_gap):

    max_rouge = 0
    ## get the indics of words
    signal_idxs = get_signal_idxs(para)
    start_idxs = [0]
    for item in signal_idxs:
        start_idxs.append(item+1)
    end_idxs = signal_idxs
    end_idxs.append(len(para))

    for j, index_start in enumerate(start_idxs):
        if get_rougel_score(para, reference, 'f') == 0.0:
            return 1, False
        if max_gap+j > len(end_idxs):
            end_point = len(end_idxs)
        else:
            end_point = max_gap + j
        for index_stop in end_idxs[j: end_point]:
            if index_start < index_stop:
                temp_score = get_rougel_score(para[index_start: index_stop], reference, 'f')
                if max_rouge < temp_score and para[index_start: index_stop]!='':
                    flag = 0
                    for ch in para[index_start: index_stop]:
                        if ch != ' ':
                            flag = 1
                            break
                    if flag == 1:
                        best_span_start = index_start
                        best_span_end   = index_stop
                        max_rouge = temp_score
                        
    if get_rougel_score(para[best_span_start: best_span_end], reference, 'f') < 0.6:
        return 1, False
    
    substring = Tokenize_string_word_level(para[best_span_start: best_span_end]) 
    word_token_para = Tokenize_string_word_level(para)
    # sent_token_para = Tokenize(para)

    index_start, index_stop = get_idx_sublist(word_token_para, substring)
    # print([index_start, index_stop])
    # print(para[best_span_start: best_span_end])
    # print(para)
    # print(reference)
    # print(max_rouge)
    # print(substring)
    # print(word_token_para)
    # print(sent_token_para)
    return [index_start, index_stop], True
    # try:
    #     return trans_idx_1dto2d(index_start, index_stop, sent_token_para), True
    # except:
    #     print(substring)
    #     print(para)
# def get_indics(para_list, ans_list):
#     indics_list = []
#     fucking_indics = []
#     for i, para in enumerate(para_list):
#         tokenized_para = Tokenize_string_word_level(para)
#         tokenized_ans = Tokenize_string_word_level(ans_list[i])
#         index_start, index_stop = get_idx_sublist(tokenized_para, tokenized_ans)
#         if index_start == -1:
#             fucking_indics.append(i)
#         else:   
#             indics_list.append([index_start, index_stop])
#     return indics_list, fucking_indics
def get_selected_span(para, selected_span):
    
    substring = Tokenize_string_word_level(selected_span)
    word_token_para = Tokenize_string_word_level(para)
    sent_token_para = Tokenize(para)
    index_start, index_stop = get_idx_sublist(word_token_para, substring)
    return [index_start, index_stop]