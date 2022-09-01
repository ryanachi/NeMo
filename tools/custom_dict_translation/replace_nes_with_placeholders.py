import argparse
import editdistance
import numpy as np
import spacy
from collections import Counter, defaultdict
from datetime import date
import os
# import pudb
from tqdm import tqdm
import mmap
import time
from datetime import timedelta
 
lang_2_spacy_pipeline = {
    "en": "en_core_web_sm",
    "de": "de_core_news_lg",
    "fr": "fr_core_news_sm",
    "ru": "ru_core_news_md",
    "es": "ru_core_news_md",
    "zh": "zh_core_web_sm",
}
 
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--src_lang", help="source language", required=True)
parser.add_argument("-b", "--tgt_lang", help="target language",required=True)
parser.add_argument("-s", "--src_file", help="source file", required=True)
parser.add_argument("-t", "--tgt_file", help="target file", required=True)
parser.add_argument("-r", "--replace", help="whether to generate new src + tgt data w/ replaced NEs", action="store_true")
parser.add_argument("-p", "--placeholders", help="whether to replace both src + tgt with $NE_x", action="store_true")
parser.add_argument("-l", "--log_file", help="log file to write results to", required=True)
parser.add_argument("-e", "--out_file_base_name")
 
args = parser.parse_args()
 
if args.src_lang in lang_2_spacy_pipeline:
    src_spacy = spacy.load(lang_2_spacy_pipeline[args.src_lang])
elif args.src_lang == "ja":
    src_spacy = spacy.blank("ja")
 
if args.tgt_lang in lang_2_spacy_pipeline:
    tgt_spacy = spacy.load(lang_2_spacy_pipeline[args.tgt_lang])
elif args.tgt_lang == "ja":
    tgt_spacy = spacy.blank("ja")
 
# one dict per parallel line    
# dicts are of the form: {NE_NUM_X: (src_translation_of_NE, tgt_translation_of_NE)}, {NE_NUM_Y: (src_trans, tgt_trans)}
NE_maps = []
diff_btw_src_NE_tgt_NE = Counter()
total_unmatched = 0
total_matched = 0
total_lines = 0
MAX_NES_IN_SENT = 20
 
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
 
def avg(a, b):
    return float(a + b) / 2
 
def merge_adjacent_NEs(word_tuples):
    n = len(word_tuples)
    out_list = []
    curr_NE = ""
    prev = -5
 
    for i in range(n):
        if word_tuples[i][1] == (prev + 1):
            curr_NE += f" {word_tuples[i][0]}"
        else:
            if i != 0: out_list.append(curr_NE)
            curr_NE = word_tuples[i][0]
        prev = word_tuples[i][1]
   
    if curr_NE != "": out_list.append(curr_NE)
 
    return out_list
   
out_src_file_placeholders = f"{args.out_file_base_name}_src_placeholders.txt"
out_src_file_eng = f"{args.out_file_base_name}_src_engNEs.txt"
out_tgt_file_placeholders = f"{args.out_file_base_name}_tgt_placeholders.txt"
out_tgt_file_eng = f"{args.out_file_base_name}_tgt_engNEs.txt"
# August 25 -- save subset of the original trainset with the sentences that are used for training
out_src_file_orig = f"{args.out_file_base_name}_src_subset_orig.txt"
out_tgt_file_orig = f"{args.out_file_base_name}_tgt_subset_orig.txt"
 
english_vocab = set(src_spacy.vocab.strings)
german_vocab = set(tgt_spacy.vocab.strings)
 
def generate_all_uppercase_runs(sent, lang='en'):
    wordcache = []
    curr_word = None
       
    split_sent = sent.replace('- ', '--').split(' ')
    STOPPING_WORDS = ['Mr', 'In']
    for idx, word in enumerate(split_sent):
        vocab = english_vocab if lang == 'en' else german_vocab
        #if word == 'The' and idx != 0: print(sent, split_sent[idx-1])
        if len(word) == 0: continue
        if word[0].isupper() and ((idx > 0 and split_sent[idx-1] not in ('.', ':')) or word.lower() not in vocab):
            if curr_word:
                curr_word += " " + word
            else:
                if word in STOPPING_WORDS:
                    wordcache.append(word)
                else:
                    curr_word = word
        else:
            if curr_word:
                wordcache.append(curr_word)
                curr_word = None
    wordcache = list(set(wordcache))
    wordcache = [w for w in wordcache if len(w) > 3]
    wordcache = [w.replace('--', '- ') for w in wordcache]
    #print(wordcache)
    #print(sent_s)
    #assert False
    return list(set(wordcache))
 
with open(args.src_file, "r") as f_s, \
  open(args.tgt_file, "r") as f_t, \
  open(out_src_file_placeholders, "w") as f_s_placeholders, \
  open(out_tgt_file_placeholders, "w") as f_t_placeholders, \
  open(out_src_file_eng, "w") as f_s_eng, \
  open(out_tgt_file_eng, "w") as f_t_eng, \
  open(out_src_file_orig, "w") as f_s_orig, \
  open(out_tgt_file_orig, "w") as f_t_orig, \
  open(args.log_file, "w") as f_log:
    total_lines = get_num_lines(args.src_file)
 
    for sent_idx, (sent_s, sent_t) in enumerate(zip(tqdm(f_s, total=391849894), f_t)):
        NEs_sent_s = generate_all_uppercase_runs(sent_s)
        NEs_sent_t = generate_all_uppercase_runs(sent_t, lang='de')
 
        curr_sent_map = {}
        leftover_NEs_s = []
        leftover_NEs_t = []
        unmatched_NEs = []
 
        assigned_source_NEs = {}
        assigned_target_NEs = set()
 
        source_NE_to_similiarities = {}
        for source_NE in NEs_sent_s:
            source_NE_to_similiarities[source_NE] = [editdistance.eval(source_NE, target_NE) for target_NE in NEs_sent_t]
 
        # pick the top scores, removing them at each point
        while len(NEs_sent_s) - len(assigned_source_NEs) and len(NEs_sent_t) - len(assigned_target_NEs):
            #print(len(NEs_sent_s), len(assigned_source_NEs), NEs_sent_s)
            # print("Assigned NEs (s)", assigned_source_NEs.keys())
            # print("Assigned NEs (t)", assigned_target_NEs)
            # print("")
            scores_to_best_similarities = {}
            for source_NE, similarities in source_NE_to_similiarities.items():
                idx, score = min([(idx, s) for idx, s in enumerate(similarities) if NEs_sent_t[idx] not in assigned_target_NEs], key=lambda kv: kv[1])
                scores_to_best_similarities[source_NE] = (idx, score)
            #print("Scores_to_best_similarities", scores_to_best_similarities)
            best_source_NE, (best_target_NE_idx, best_score) = min(scores_to_best_similarities.items(), key = lambda kv: kv[1][1])
            if best_score > len(best_source_NE)//2: break
            best_target_NE = NEs_sent_t[best_target_NE_idx]
            assigned_source_NEs[best_source_NE] = best_target_NE
            assigned_target_NEs.add(best_target_NE)
            del source_NE_to_similiarities[best_source_NE]
 
        for s, t in assigned_source_NEs.items():
            if s != t:
                print(f"Sentence {sent_idx}: assigned {s} to {t} \n")
                print(f"{sent_s} {sent_t} \n")
                f_log.write(f"Sentence {sent_idx}: assigned {s} to {t} \n")
                f_log.write(f"{sent_s} {sent_t} \n")
 
 
 
        # REPLACEMENT CODE
        if args.replace:
            # Aug 5 modification: only write if there's at least one replacement going on
 
            if len(assigned_source_NEs) and len(assigned_source_NEs) <= MAX_NES_IN_SENT:
                # UNCOMMENT THIS!!!!!
 
                # reversed_assigned_source_NEs = {v: k for k, v in assigned_source_NEs.items()}
                # englished_sent_t = sent_t
                # placeholder_sent_t = sent_t
                # placeholder_sent_s = sent_s
                # for i, (k, v) in enumerate(reversed_assigned_source_NEs.items()):
                #     englished_sent_t = englished_sent_t.replace(k, v)
                #     placeholder = f"<NE{i}>"
                #     placeholder_sent_t = placeholder_sent_t.replace(k, placeholder)
                #     placeholder_sent_s = placeholder_sent_s.replace(v, placeholder)
 
 
                # f_t_eng.write(f"{englished_sent_t}")
                # f_s_eng.write(f"{sent_s}")
                # f_t_placeholders.write(f"{placeholder_sent_t}")
                # f_s_placeholders.write(f"{placeholder_sent_s}")
                f_s_orig.write(f"{sent_s}")
                f_t_orig.write(f"{sent_t}")
 
 

