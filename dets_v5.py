
import pandas as pd
import numpy as np
import csv
from numpy import median
import re
import getpass
import collections
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModel, AutoModelForCausalLM
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util
import torch
from analytics.bamboo import Bamboo as bb
from bento import fwdproxy
import nltk
from nltk_data.init import init_nltk_data
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import time
from spacy.lang.en import English
from langchain.chat_models.metagen import MetaGenChat
import copy
from pvc2 import TableParameters
import pvc2
from nltk.stem import PorterStemmer
ps = PorterStemmer()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Not showing warning from Transformers
from transformers.utils import logging
logging.set_verbosity(40)

ctx = pvc2.context(namespace='hr', oncall='wfs')


def flatten_list(input_list):
    if isinstance(input_list, list):
        temp = []
        for elem in input_list:
            temp.extend(flatten_list(elem))
        return temp
    else:
        return [input_list]



def embedding_same_group(embedding_vector, word1, word2):
    for sublist in embedding_vector:
        found = 0
        if isinstance(sublist, list):
            for ww in sublist:
                if ww.startswith(word1):
                    found = found + 1
                if ww.startswith(word2):
                    found = found + 1
        else:
            if sublist.startswith(word1):
                found = found + 1
            if sublist.startswith(word2):
                found = found + 1
        if found == 2:
            return found
            break
    return found




def top_keywords(data_comment, num_keyword=40, Custom_stopwords=False, Stopwords_path='/fbsource/fbcode/paws_nlp/words_list/stopwords'):
    # get the "comment" column
    textual_data = data_comment

    if Custom_stopwords == False:
        path_words = '/home/' + str(getpass.getuser()) + Stopwords_path
    else:
        path_words = Stopwords_path

    # Stopwords
    stopwords = set(line.strip() for line in open(path_words))
    #stopwords = set(line.strip() for line in open('stopwords.txt'))
    stopwords = stopwords.union(set(['mr','mrs','one','two','said']))

    wordcount = {}
    unique = set(textual_data)

    # To eliminate duplicates, remember to split by punctuation, and use case demiliters.
    for word in textual_data.lower().split():
        word = word.replace(".","")
        word = word.replace(",","")
        word = word.replace(":","")
        word = word.replace("\"","")
        word = word.replace("!","")
        word = word.replace("â€œ","")
        word = word.replace("â€˜","")
        word = word.replace("*","")
        if word not in stopwords:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1

    Top_Keywords = []
    word_counter = collections.Counter(wordcount)
    for word, count in word_counter.most_common(num_keyword):
        Top_Keywords.append(word)
        print(word, count)

    return Top_Keywords

# Read csv file
df = pd.read_excel("/data/users/golest/Competitor-Employees-Q1-2024-Data.xlsx")

num_unique = data_df['employee_id'].nunique()
print(num_unique)

# --------------------------------------
# Setting Parameters
# Custom Embedding Parameters
Min_sentence_len = 5        # Minimum sentence length
Max_sentence_len = 1000       # Maximum sentence length
Stopwords_path='/fbsource/fbcode/paws_nlp/words_list/stopwords'
CustomEmbedding_flag = True       # False: if you don't want to use Custome Embedding
Custom_stopwords = False     # True: if you want to provide the path to your own Embedding vector
Custome_Embedding_size = 80  # *** Recommended: 50 or higher

# Modeling Parameters
considered_sentiments = ['negative', 'positive', 'neutral']
#considered_sentiments = ['positive']
sentiment_threshold = 0.1                      # Filtering the sentence with sentiment scores below this threshold
sentence_quality_threshold = 0.05              # Filtering the sentence with English quality scores below this threshold (the higher the better)
sentence_similarity_threshold = 0.35          # *** Filtering the sentence with similarity scores below this threshold (the higher the better)
sentence_similarity_threshold_absolute = 0.6   
sentence_similarity_default = 0.4
#
cluster_core_size = 0
top_percentile = 0.2
cluster_best_candidate_probability = 0.8
cluster_confidence_length = 20 # *** number of sentences in each cluster/key point to be checked with

llama_shorten_sentence_threshold = 9000 # number of characters
llama_size_output = 160 # in characters
llama_size_output_keypoint = 230
summarization_length_threshold = 0.007 # ***in percentage. The range is from 0 to 0.1       default for small comments: 0.0006     default for large comments: 0.005

# Themes Parameters
number_core_themes = 0  # the minimum number of core themes to have
similarity_waiver_for_high_overalp = 0.02   # Default for large N: 0.07     default for large N: 0.14       higher value mean less restriction on merging keypoints to themes leading to less number of themes.
theme_confidence_length = 20 # *** number of key points in each key point/cluster to be checked with

#---------------------------------------------------------
#    Phase 1
#---------------------------------------------------------
print("1- Initiating LLMs")

device_sm = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device_sm)
device_numeric = 0 if torch.cuda.is_available() else -1
print(device_numeric)
# Sentence Similarity
model_mnli = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v4', device=device_sm, max_length=512)
#model_mnli = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v3-light', device=device_sm)
#model_mnli = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v2', device=device_sm)
# model_mnli = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v1', device=device_sm)

nltk.download('punkt')

# sentiment transformer
#sentiment_classifier = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment", device=device_numeric)
sentiment_classifier = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device_numeric)

 # sentence quality transformer
tokenizer_sentence_quality = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
model_sentence_quality = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").to(device_sm)

tokenizer_llm = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
model_llm = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")

#jjj = top_keywords(data_df.comment.to_string(), Custome_Embedding_size, Custom_stopwords)
#print(jjj)

#---------------------------------------------------------
#    Phase 2
#---------------------------------------------------------

print("2- Sort Sentences Based on Quality (RoBERTa_CoLA)")
# Start from Scratch
employee_id = []
comment_list = []
sentence_list = []
sentence_size = []
score_list = []
Nb_match_list = []
sentiment_label_list = []
sentiment_score_list = []

for index1, row1 in data_df.iterrows():
    if index1 % 1000 == 0:
        print(index1)
    whole_comment = row1['comment'].replace(',', '')
    comment_sentences_list = nltk.tokenize.sent_tokenize(whole_comment)

    for sent in comment_sentences_list:
        sent_text = sent
        sent_text = sent_text.replace('.', '')
        sent_text = sent_text.replace('!', '')
        sent_text = sent_text.replace('?', '')
        if(len(sent_text) > Min_sentence_len and len(sent_text) < Max_sentence_len ):
            # Checking English quality of a sentence
            # quality_score = score(sent_text)[0][1].item()
            quality_score = 0.2222

            employee_id.append(str(row1['employee_id']))
            sentence_list.append(sent_text)
            sentence_size.append(len(sent_text))
            score_list.append(quality_score)
            comment_list.append(row1['comment'])
    
data_df_sorted1 = pd.DataFrame(
    {'employee_id': employee_id,
     'comment': comment_list,
     'sentence': sentence_list,
     'sentence_quality_score': score_list,
     'sentence_size': sentence_size
    })

#data_df_sorted = data_df_sorted.sort_values('sentence_quality_score', ascending=False) # Highest English Quality to Lowest
data_df_sorted1 = data_df_sorted1.sort_values('sentence_size', ascending=True)  # Shortes to Longest
#data_df_sorted = data_df_sorted.sort_values('sentence_size', ascending=False) # Longest to Shortest
data_df_sorted1 = data_df_sorted1.reset_index(drop=True)

print("Number of sentences: ", len(data_df_sorted1))
data_df_sorted1_1 = data_df_sorted1[(data_df_sorted1['sentence_size'] >= 40) & (data_df_sorted1['sentence_size'] <= 80)]
data_df_sorted1_2 = data_df_sorted1[data_df_sorted1['sentence_size'] < 40]
data_df_sorted1_3 = data_df_sorted1[data_df_sorted1['sentence_size'] > 80]
frames = [data_df_sorted1_1, data_df_sorted1_2, data_df_sorted1_3]
 
data_df_sorted = pd.concat(frames)

# Number of unique employees
u_employees = len(pd.unique(data_df_sorted['employee_id']))
print('Number of unique employees: ', u_employees)

#---------------------------------------------------------
#    Phase 3
#---------------------------------------------------------
print("3- Clustering Based on DeTS")

# Using Custome Embedding
dict_custom_embedding1 = {}
if CustomEmbedding_flag:
    #custom_embedding1 = top_keywords(data_df.comment.to_string(), Custome_Embedding_size, Custom_stopwords)
    #custom_embedding_grouped = custom_embedding1

    custom_embedding_grouped = [['ai', 'superintelligence', 'talent'], ['direction', 'investment', 'leadership', 'vision', 'strategy'],
                                ['wearables', 'glasses', 'smart', 'metaverse', 'products']]
    
    no_stem_list = ['direction', 'decisions', 'investment', 'vision', 'products']

    custom_embedding1 = flatten_list(custom_embedding_grouped)

    print(custom_embedding1)
    for index, element in enumerate(custom_embedding1):
        if element in no_stem_list or element[-1]=='y':
            dict_custom_embedding1[element] = 0
        else:
            dict_custom_embedding1[ps.stem(element)] = 0

print(dict_custom_embedding1)
# Start from Scratch
employee_id = []
sentence_list = []
comment_list = []
score_list = []
Nb_match_list = []
sentiment_label_list = []
sentiment_score_list = []

start_time = time.time()

num_cluster = -1
clusters      = []
clusters_dict = []

for index1, row1 in data_df_sorted.iterrows():
    if index1 % 1000 == 0:
        print(index1)
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60
        print("Elapsed time (minutes): ", elapsed_time)
        
    employee_id = row1['employee_id']

    sent_text_original = row1['sentence']
    if len(sent_text_original) >= llama_shorten_sentence_threshold:
        short_summary_prompt = f"""
        Task: Provide a concise and 10-words summary of the main points in this text about working condition: '{sent_text_original}'.
        """
        messages = [
            {"role": "user", "content": short_summary_prompt},
        ]
        inputs = tokenizer_llm.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model_llm.device)

        outputs = model_llm.generate(**inputs, max_new_tokens=40)
        sent_text = (tokenizer_llm.decode(outputs[0][inputs["input_ids"].shape[-1]:]))[:-10]
    else:
        sent_text = sent_text_original

    quality_score = row1['sentence_quality_score']
    comment_text = row1['comment']

    if(len(sent_text) > Min_sentence_len and len(sent_text) < Max_sentence_len):
        if  quality_score > sentence_quality_threshold:
            #
            result = sentiment_classifier(sent_text)
            if result[0]["label"] == 'LABEL_2' or result[0]["label"] == 'positive':
                sentiment_label = "positive"
            elif result[0]["label"] == 'LABEL_1' or result[0]["label"] == 'neutral':
                    sentiment_label = "neutral"
            elif result[0]["label"] == 'LABEL_0' or result[0]["label"] == 'negative':
                sentiment_label = "negative"
            else:
                print("No sentiment???")
                sentiment_label ="na"
                result[0]["score"] = 0
            #
            if (sentiment_label in considered_sentiments) and result[0]["score"] >= sentiment_threshold: # if sentiment is positive or negative
                dict_custom_embedding_tmp = copy.copy(dict_custom_embedding1)
                topwords = []
                res = sent_text.lower().split() 
                for word1 in dict_custom_embedding1:
                    for word2 in res:
                        #if ps.stem(word1) in sent_text.lower():
                        #ticker = re.findall(r'\b[word1]', sent_text.lower()
                        if word2.startswith(word1):
                            dict_custom_embedding_tmp[word1] = dict_custom_embedding_tmp.get(word1, 0) + 1
                            topwords.append(word1)
                #
                if num_cluster > -1 and (len(topwords) > 0 or CustomEmbedding_flag == False):
                    similar_cluster_score = 0
                    max_score = 0
                    max_cluster = 0
                    i_cluster_max = 0
                    for idx, i_cluster in enumerate(clusters):
                        total_sum = 0
                        n_word_match = 0
                        ev_average_values = sum(clusters_dict[idx].values()) / len(clusters_dict[idx]) 
                        for wws in topwords:
                            total_sum = total_sum + (clusters_dict[idx][wws]/ev_average_values)
                            #total_sum = total_sum + clusters_dict[idx][wws]
                            if clusters_dict[idx][wws] > 0:
                                n_word_match = n_word_match + 1
                        #total_sum = total_sum / (1+(n_word_match * 10))
                        if total_sum > similar_cluster_score:
                            similar_cluster_score = total_sum
                            max_cluster = idx
                            i_cluster_max = i_cluster
                    #
                    if similar_cluster_score > 0:
                        cluster_sentences = [item[2] for item in i_cluster_max[:cluster_confidence_length]]
                        new_sent_list = [sent_text] * len(cluster_sentences)
                        pairs = zip(new_sent_list, cluster_sentences)
                        list_pairs=list(pairs)
                        sim_scores = (model_mnli.predict(list_pairs, show_progress_bar=False)).tolist()
                        max_list = max(sim_scores)
                        if max_list > max_score:
                                max_score = max_list
                        #
                        if max_score >= (sentence_similarity_threshold):
                            clusters[max_cluster].append((employee_id, comment_text, sent_text, quality_score, max_score, max_cluster, sent_text_original))
                            for wws in topwords:
                                clusters_dict[max_cluster][wws] = clusters_dict[max_cluster][wws] + 1
                        else:
                            num_cluster = num_cluster + 1
                            clusters.extend([[(employee_id, comment_text, sent_text, quality_score, 0.5, num_cluster, sent_text_original)]])
                            clusters_dict.extend([copy.copy(dict_custom_embedding_tmp)])
                    else:
                        num_cluster = num_cluster + 1
                        clusters.extend([[(employee_id, comment_text, sent_text, quality_score, 0.5, num_cluster, sent_text_original)]])
                        clusters_dict.extend([copy.copy(dict_custom_embedding_tmp)])
                    #
                elif num_cluster == -1 and len(topwords) > 0:
                    num_cluster = num_cluster + 1
                    clusters_dict.extend([copy.copy(dict_custom_embedding_tmp)])
                    clusters.extend([[(employee_id, comment_text, sent_text, quality_score, sentence_similarity_default, num_cluster, sent_text_original)]])

end_time = time.time()
elapsed_time = (end_time - start_time)/60
print("Elapsed Time (minutes): ", elapsed_time)
print("Number of Clusters: ", num_cluster)

#---------------------------------------------------------
#    Phase 4
#---------------------------------------------------------
print("4- Forming the Final Clusters + Llama2 Summaries")

# Forming the Clusters
cluster_index = []
cluster_top_candidate = []
cluster_summary = []
cluster_words = []
cluster_words_max = []
cluster_size = []

df_employee_id = []
df_question_group_id = []
df_response_date = []
df_survey_period = []
df_pyx_tag = []
df_comment = []
df_sentence = []
df_sentence_original = []
df_similarity_score = []
df_cluster_index = []

print("Accepted cluster length for summarization: ", (summarization_length_threshold * len(data_df_sorted)))

for idx, i_cluster in enumerate(clusters):
    if idx % 1000 == 0:
        print(idx)
    cluster_index.append(i_cluster[0][5])
    #cluster_best_candidate = max(i_cluster, key=lambda x: (0.01*x[3]+3*x[4])) # english_qualiy + 2 * sentence similarity
    cluster_best_candidate = max(i_cluster, key=lambda x: (1/(x[4]%cluster_best_candidate_probability))) # english_qualiy + 2 * sentence similarity
    cluster_top_candidate.append(cluster_best_candidate[2])
    cluster_words.append(clusters_dict[idx])
    #
    all_sentneces = ""
    
    for i_sent in i_cluster:
        all_sentneces = all_sentneces + i_sent[2] + '. '

        df_employee_id.append(i_sent[0])
        df_comment.append(i_sent[1])
        df_sentence.append(i_sent[2])
        df_sentence_original.append(i_sent[6])
        df_similarity_score.append(i_sent[4])
        df_cluster_index.append(i_cluster[0][5])
    #summarization
    if len(all_sentneces) > 15000:
        all_sentneces_truncated = all_sentneces[:15000]
    else:
        all_sentneces_truncated = all_sentneces
    #
    if len(i_cluster) > (summarization_length_threshold * len(data_df_sorted)):
        summary_prompt = f"""
            Task: Summarize this text: '{all_sentneces_truncated}'.

            Output Format:
            - Provide an output with maximum of '{llama_size_output_keypoint}' Characters.

            Begin your response with 'this key point describes '.
            """
        
        messages = [
            {"role": "user", "content": summary_prompt},
        ]
        inputs = tokenizer_llm.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model_llm.device)

        outputs = model_llm.generate(**inputs, max_new_tokens=40)
        summary_out = (tokenizer_llm.decode(outputs[0][inputs["input_ids"].shape[-1]:]))[:-10]
    else:
        summary_out = cluster_best_candidate[2]  # replace summary with the best candidate
    cluster_summary.append(summary_out)
    cluster_size.append(len(i_cluster))


df_cluster = pd.DataFrame(
    {'cluster_index': cluster_index,
     'cluster_top_candidate': cluster_top_candidate,
     'cluster_summary': cluster_summary,
     'cluster_words': cluster_words,
     'cluster_size': cluster_size
    })

df_all_data = pd.DataFrame(
    {
        'employee_id': df_employee_id,
        'comment': df_comment,
        'sentence': df_sentence,
        'sentence_original': df_sentence_original,
        'similarity_score': df_similarity_score,
        'cluster_index': df_cluster_index
    }
)

df_cluster_sorted = df_cluster.sort_values('cluster_size', axis=0, ascending=False,inplace=False)
df_cluster_sorted = df_cluster_sorted.reset_index(drop=True)

df_all_data_clusters = pd.merge(df_all_data, df_cluster, on='cluster_index',  how='left')
#df_all_data_clusters = df_all_data_clusters.drop(['cluster_size'], axis=1)

# Number of unique employees
nn = len(pd.unique(df_all_data_clusters['employee_id']))
print("Number of employees with a keypoint: ", nn)

#print(df_cluster_sorted.head())
#---------------------------------------------------------
#    Phase 5
#---------------------------------------------------------
print("5- Themes: Merging Keypoints into Themes")

num_theme = -1
themes = []
themes_dict = []
remaining_cluster = []

index2 = 0
for row in df_cluster_sorted.itertuples():
    index2 = index2 + 1
    if index2 % 1000 == 0:
        print(index2)
    sent_text = nltk.tokenize.sent_tokenize(row[3])[0]
    dict_startwords_tmp = copy.copy(row[4])
    topwords = []

    if num_theme >= number_core_themes:
        similar_theme_score = 0
        max_score = 0
        max_theme = 0
        i_theme_max = 0
        for idx, i_theme in enumerate(themes):
            total_sum = 0
            #
            ev_average_values = sum(themes_dict[idx].values()) / len(themes_dict[idx])
            max_word_current = max(dict_startwords_tmp, key=dict_startwords_tmp.get)
            max_word_theme = max(themes_dict[idx], key=themes_dict[idx].get)
            for k in dict_startwords_tmp:
                #if dict_startwords_tmp[k] > 0 and themes_dict[idx][k] > 0:
                total_sum = total_sum + (themes_dict[idx][k]/ev_average_values)
                    #total_sum = total_sum + (themes_dict[idx][k] / dict_startwords_tmp[k])
                    #total_sum = total_sum + 1
            if total_sum > similar_theme_score and embedding_same_group(custom_embedding_grouped, max_word_current, max_word_theme) == 2: #max_word_current == max_word_theme:
                similar_theme_score = total_sum
                max_theme = idx
                i_theme_max = i_theme
        #
        if similar_theme_score > 0:
            theme_sentences = [item[2] for item in i_theme_max[:theme_confidence_length]]
            new_sent_list = [sent_text] * len(theme_sentences)
            pairs = zip(new_sent_list, theme_sentences)
            list_pairs=list(pairs)
            sim_scores = (model_mnli.predict(list_pairs, show_progress_bar=False)).tolist()
            max_list = max(sim_scores)
            if max_list > max_score:
                    max_score = max_list

            if max_score >= (sentence_similarity_threshold - (similarity_waiver_for_high_overalp * similar_theme_score*5)): # 0.1 decrease for every ratio of 5X common words  (0.1 = 0.02 * 5)
                themes[max_theme].append((row[1], row[3], sent_text, row[4], row[5], max_score, max_theme))#cluster_index,cluster_top_candidate,cluster_summary,cluster_words,cluster_size
                for key in dict_startwords_tmp:
                    themes_dict[max_theme][key] = themes_dict[max_theme][key] + dict_startwords_tmp[key]
            else:
                num_theme = num_theme + 1
                themes.extend([[(row[1], row[3], sent_text, row[4], row[5], 0.2, num_theme)]])
                themes_dict.extend([copy.copy(dict_startwords_tmp)])
        else:
            remaining_cluster.append(row[1])
    else:
        num_theme = num_theme + 1
        themes.extend([[(row[1], row[3], sent_text, row[4], row[5], 0.2, num_theme)]])
        themes_dict.extend([copy.copy(dict_startwords_tmp)])


print("----- Phase 2 ------")
index2 = 0

for ii in remaining_cluster:
    cluster_columns = df_cluster_sorted.loc[df_cluster_sorted['cluster_index'] == ii]
    index2 = index2 + 1
    if index2 % 1000 == 0:
        print(index2)
    for row2 in cluster_columns.itertuples():
        sent_text = nltk.tokenize.sent_tokenize(row2[3])[0]
        dict_startwords_tmp = copy.copy(row2[4])
        topwords = []

        if num_theme > number_core_themes:
            similar_theme_score = 0
            max_score = 0
            max_theme = 0
            i_theme_max = 0
            for idx, i_theme in enumerate(themes):
                total_sum = 0
                #
                ev_average_values = sum(themes_dict[idx].values()) / len(themes_dict[idx])
                max_word_current = max(dict_startwords_tmp, key=dict_startwords_tmp.get)
                max_word_theme = max(themes_dict[idx], key=themes_dict[idx].get)
                for k in dict_startwords_tmp:
                    #if dict_startwords_tmp[k] > 0 and themes_dict[idx][k] > 0:
                    total_sum = total_sum + (themes_dict[idx][k]/ev_average_values)
                        #total_sum = total_sum + (themes_dict[idx][k] / dict_startwords_tmp[k])
                        #total_sum = total_sum + 1
                if total_sum > similar_theme_score and embedding_same_group(custom_embedding_grouped, max_word_current, max_word_theme) == 2: #max_word_current == max_word_theme:
                    similar_theme_score = total_sum
                    max_theme = idx
                    i_theme_max = i_theme
            #
            if similar_theme_score > 0:
                theme_sentences = [item[2] for item in i_theme_max[:theme_confidence_length]]
                new_sent_list = [sent_text] * len(theme_sentences)
                pairs = zip(new_sent_list, theme_sentences)
                list_pairs=list(pairs)

                # Pad the input tensor to the expected size
                # max_length = 580
                # padded_pairs = []
                # for pair in list_pairs:
                #     padded_pair = (pair[0] + [''] * (max_length - len(pair[0])), pair[1])
                #     padded_pairs.append(padded_pair)
                # sim_scores = (model_mnli.predict(padded_pairs, show_progress_bar=False)).tolist()



                sim_scores = (model_mnli.predict(list_pairs, show_progress_bar=False)).tolist()
                max_list = max(sim_scores)
                if max_list > max_score:
                        max_score = max_list

                if max_score >= (sentence_similarity_threshold - (similarity_waiver_for_high_overalp * similar_theme_score*5)): # 0.1 decrease for every ratio of 5X common words  (0.1 = 0.02 * 5)
                    themes[max_theme].append((row2[1], row2[3], sent_text, row2[4], row2[5], max_score, max_theme))#cluster_index,cluster_top_candidate,cluster_summary,cluster_words,cluster_size
                    for key in dict_startwords_tmp:
                        themes_dict[max_theme][key] = themes_dict[max_theme][key] + dict_startwords_tmp[key]
                else:
                    num_theme = num_theme + 1
                    themes.extend([[(row2[1], row2[3], sent_text, row2[4], row2[5], 0.2, num_theme)]])
                    themes_dict.extend([copy.copy(dict_startwords_tmp)])
            else:
                num_theme = num_theme + 1
                themes.extend([[(row2[1], row2[3], sent_text, row2[4], row2[5], 0.2, num_theme)]])
                themes_dict.extend([copy.copy(dict_startwords_tmp)])
        else:
            num_theme = num_theme + 1
            themes.extend([[(row2[1], row2[3], sent_text, row2[4], row2[5], 0.2, num_theme)]])
            themes_dict.extend([copy.copy(dict_startwords_tmp)])


print("Number of themes: ", len(themes))

#---------------------------------------------------------
#    Phase 6
#---------------------------------------------------------
print("6- Grouping Key Points into Themes")

theme_index = []
theme_summary = []
theme_size = []
theme_embedding_vector = []

df_cluster_id = []
df_mapped_theme_id = []


for idx, i_theme in enumerate(themes):
    theme_index.append(i_theme[0][6])
    theme_embedding_vector.append(themes_dict[idx])
    #
    theme_size_int = 0
    all_sentneces = ""
    first_cluster_summary = ""
    for i_sent in i_theme:
        all_sentneces = all_sentneces + i_sent[2] + '. '
        if theme_size_int == 0: # cluster summary of first cluster in the theme
            first_cluster_summary = i_sent[1] 
        theme_size_int = theme_size_int + i_sent[4]

        df_cluster_id.append(i_sent[0])
        df_mapped_theme_id.append(i_sent[6])

    #
    #summarization
    if len(all_sentneces) > 15000:
        all_sentneces_truncated = all_sentneces[:15000]
    else:
        all_sentneces_truncated = all_sentneces
    #
    summary_prompt = f"""
        Provide a general summary for this text: '{all_sentneces_truncated}'.

        Output Format:
        - The summary must be only the title.
        - The maximum length of the output should be 120 Characters.

        """
    #summary_out = llmchat.predict(summary_prompt)
    #
    #theme_summary.append(summary_out)
    theme_summary.append(first_cluster_summary)
    #
    theme_size.append(theme_size_int)


df_theme = pd.DataFrame(
    {'theme_id': theme_index,
     'theme_summary_1st_cluster': theme_summary,
     'theme_embedding_vector': theme_embedding_vector,
     'theme_size': theme_size
    })

df_theme_mapped_cluster_data = pd.DataFrame(
    {
        'cluster_index': df_cluster_id,
        'theme_id': df_mapped_theme_id
    }
)

df_clusters_themes = pd.merge(df_theme_mapped_cluster_data, df_theme, on='theme_id',  how='left')
df_all_data_clusters_themes = pd.merge(df_all_data_clusters, df_clusters_themes, on='cluster_index',  how='left')
df_all_data_clusters_themes = df_all_data_clusters_themes.drop(['theme_size'], axis=1)

#---------------------------------------------------------
#    Phase 7
#---------------------------------------------------------
print("7- Forming the Final Themes + Llama2 Summaries")

theme_index2 = []
theme_summary2 = []
for i in range(len(themes)):
    #print(i)
    theme_index2.append(i)
    #df_all_data_clusters_themes
    ddd = df_all_data_clusters_themes.loc[df_all_data_clusters_themes['theme_id'] == i]
    all_sentneces = ""
    for index, row in ddd.iterrows():
        all_sentneces = all_sentneces + row['sentence'] + '. '
    #summarization
    if len(all_sentneces) > 15000:
        all_sentneces_truncated = all_sentneces[:15000]
    else:
        all_sentneces_truncated = all_sentneces
    #
    #
    summary_prompt = f"""
            Generate a concise title that captures the overall sentiment expressed for Collection of comments.

            Output Format:
            - The output must be only the title.
            - The title's length should be around of {llama_size_output} Characters.

            Comments Collection are: {all_sentneces_truncated}
            """



    # summary_prompt = f"""
    #         Task: Summarize this text: {all_sentneces_truncated}.

    #         Output Format:
    #         - Provide an output with maximum of {llama_size_output} Characters.
    #         - provide percentages for each sub-section of the summary.

    #         Begin your response with 'this theme describes '.
    #         """

    # summary_prompt = f"""
    #         Task: Summarize this text: {all_sentneces_truncated}.

    #         Output Format:
    #         - Provide percentages for each sub-section of the summary.
    #         - The sub-section's title should NOT be based on the sentiment and should not include words like Positive, Negative, Neutral.
    #         - The sub-section's title should be like short phrases based on the content of the comments.
    #         - Provide 3 most relevant comments for each sub-section.
    #         - Exluding relevant comments, the summary and the sub-sections should be maximum of 300 Characters.

    #         Begin your response with 'this theme describes '.
    #         """
            #- The overall output should be maximum of {llama_size_output} Characters.
    
    messages = [
        {"role": "user", "content": summary_prompt},
    ]
    inputs = tokenizer_llm.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_llm.device)

    outputs = model_llm.generate(**inputs, max_new_tokens=40)
    summary_out = (tokenizer_llm.decode(outputs[0][inputs["input_ids"].shape[-1]:]))[:-10]
    
    theme_summary2.append(summary_out)

#
df_theme_summary = pd.DataFrame(
    {
        'theme_id': theme_index2,
        'theme_summary': theme_summary2
    }
)
#
df_all_data2 = pd.merge(df_all_data_clusters_themes, df_theme_summary, on='theme_id',  how='left')

#
df_all_data2['theme_summary'] = df_all_data2['theme_summary'].str.replace("'", '"')
df_all_data2['cluster_words'] = df_all_data2['cluster_words'].astype(str)
df_all_data2['theme_embedding_vector'] = df_all_data2['theme_embedding_vector'].astype(str)

print("number of rows in final dataframe: ", len(df_all_data2))

top_themes = df_all_data2.groupby(['theme_summary','theme_embedding_vector']).size().sort_values(ascending=False).reset_index()


# Saving the Results
#df_all_data_clusters.to_csv("Clustering_DeTS_QIPT_Q29_Wave2_V2.csv")
#df_all_data2.to_excel("Clustering_DeTS_pulse_2024H1_career_0_v4.xlsx")
df_all_data2.to_excel("DeTS_MP_2024H2_CrossEncoder_Management_v4_0.35.xlsx")


#---------------------------------------------------------
#    Phase 7
#---------------------------------------------------------
print("7- Forming the Final Themes + Llama2 Summaries")

theme_index2 = []
theme_summary2 = []
for i in range(len(themes)):
    #print(i)
    theme_index2.append(i)
    #df_all_data_clusters_themes
    ddd = df_all_data_clusters_themes.loc[df_all_data_clusters_themes['theme_id'] == i]
    all_sentneces = ""
    for index, row in ddd.iterrows():
        all_sentneces = all_sentneces + row['sentence'] + '. '
    #summarization
    if len(all_sentneces) > 15000:
        all_sentneces_truncated = all_sentneces[:15000]
    else:
        all_sentneces_truncated = all_sentneces
    #
    #
    summary_prompt = f"""
            Generate a concise title that captures the overall sentiment expressed for Collection of comments.

            Output Format:
            - The title's length should be around of {llama_size_output} Characters.
            - The output must be only the title.

            Comments Collection are: {all_sentneces_truncated}
            """

    #- add two corresponding percentages for the main 2 sub-sections without changing the current wording when there is a mixed title.


    # summary_prompt = f"""
    #         Task: Summarize this text: {all_sentneces_truncated}.

    #         Output Format:
    #         - Provide an output with maximum of {llama_size_output} Characters.
    #         - provide percentages for each sub-section of the summary.

    #         Begin your response with 'this theme describes '.
    #         """

    # summary_prompt = f"""
    #         Task: Summarize this text: {all_sentneces_truncated}.

    #         Output Format:
    #         - Provide percentages for each sub-section of the summary.
    #         - The sub-section's title should NOT be based on the sentiment and should not include words like Positive, Negative, Neutral.
    #         - The sub-section's title should be like short phrases based on the content of the comments.
    #         - Provide 3 most relevant comments for each sub-section.
    #         - Exluding relevant comments, the summary and the sub-sections should be maximum of 300 Characters.

    #         Begin your response with 'this theme describes '.
    #         """
            #- The overall output should be maximum of {llama_size_output} Characters.
    
    
    messages = [
        {"role": "user", "content": summary_prompt},
    ]
    inputs = tokenizer_llm.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_llm.device)

    outputs = model_llm.generate(**inputs, max_new_tokens=40)
    
    
    summary_out_1 = (tokenizer_llm.decode(outputs[0][inputs["input_ids"].shape[-1]:]))[:-10]


    summary_prompt_2nd_round = f"""
            - Provide two percentages for the positive and negative part of this title: {summary_out_1}.

            Output Format:
            - Provide the two percentages only if the title is mixed
            - The percentages should be based on the comments that the title is based on
            - The title is based on these comments: {all_sentneces_truncated}

            """


    messages = [
        {"role": "user", "content": summary_prompt_2nd_round},
    ]
    inputs = tokenizer_llm.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_llm.device)

    outputs = model_llm.generate(**inputs, max_new_tokens=40)
    
    summary_out = (tokenizer_llm.decode(outputs[0][inputs["input_ids"].shape[-1]:]))[:-10]

    print('========================')
    print(summary_out_1)
    print("-----")
    print(summary_out)

    theme_summary2.append((summary_out_1 + summary_out))
    #theme_summary2.append((summary_out_1))

#
df_theme_summary = pd.DataFrame(
    {
        'theme_id': theme_index2,
        'theme_summary': theme_summary2
    }
)
#
df_all_data2 = pd.merge(df_all_data_clusters_themes, df_theme_summary, on='theme_id',  how='left')

#
df_all_data2['theme_summary'] = df_all_data2['theme_summary'].str.replace("'", '"')
df_all_data2['cluster_words'] = df_all_data2['cluster_words'].astype(str)
df_all_data2['theme_embedding_vector'] = df_all_data2['theme_embedding_vector'].astype(str)

print("number of rows in final dataframe: ", len(df_all_data2))

top_themes = df_all_data2.groupby(['theme_summary','theme_embedding_vector']).size().sort_values(ascending=False).reset_index()






