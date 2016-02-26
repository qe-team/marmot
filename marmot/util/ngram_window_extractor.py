#!/usr/bin/env python
#encoding: utf-8

'''
@author: Chris Hokamp
@contact: chris.hokamp@gmail.com
'''


# this only finds the first instance of the token in the sentence (see the idx= keyword arg of the extract_window function)
def locate_token(token, sentence):
    try:
        i = sentence.index(token)
        return i
    except ValueError:
        return -1


# window_size is the window on both left and right of the token of interest
def extract_window(token_list, token, window_size=1, with_token=True, idx=None):
    index = idx or locate_token(token, token_list)
    if index != -1:
        if with_token:
            context_args = [token_list, token, window_size, index]
            window = left_context(*context_args) + [token] + right_context(*context_args)
            return window
        else:
            context_args = [token_list, token, window_size, index]
            window = left_context(*context_args) + right_context(*context_args)
            return window
    return None


def left_context(token_list, token, context_size=1, idx=None):
    index = idx or locate_token(token, token_list)
    left_window = []
    if index != -1:
        for i in range(index-context_size, index):
            if i < 0:
#                left_window.append('_START_')
                left_window.append('<s>')
            else:
                left_window.append(token_list[i])
    return left_window


def right_context(token_list, token, context_size=1, idx=None):
    index = idx or locate_token(token, token_list)
    right_window = []
    if index != -1:
        for i in range(index+1, index+context_size+1):
            if i > len(token_list)-1:
#                right_window.append('_END_')
                right_window.append('</s>')
            else:
                right_window.append(token_list[i])
    return right_window

