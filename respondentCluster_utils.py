SEED = 100

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.tools as tls

from wordcloud import WordCloud

from IPython.display import HTML as html_print
from IPython.core.display import display

import spacy
from spacy.lang.en import English

# nlp = spacy.load('en_core_web_sm')
global_nlp = spacy.load("en_core_web_sm")

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords as nltl_sw
# from textblob import TextBlob


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

import logging

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


ilist_color_0 = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                 '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                 '#000075', '#808080', '#ffffff', '#000000']

ilist_color_1 = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                 '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075',
                 '#808080', '#ffffff', '#000000']

NA_DATA = 'noData'
iALLOWED_POSTAGS = ['NOUN', 'ADJ', 'VERB', 'ADV']
iCOLUMN_NLP = 'nlp_data'

# SEED = 100

col_to_predict = 'is_duplicate'
col_ts = 'Timestamp'
col_user = 'Username'
col_nuAns = 'nuAns'
col_puAns = 'puAns'
col_ques = 'ques'
col_all_fques = 'all_fques'

col_text = 'text'
col_num_na = 'num_na'
col_perc_na = 'perc_na'
col_sent = 'sentiment_score'
col_num_words = 'num_words'
col_sent_type = 'sent_type'

col_topic_num = 'topic_num'
col_topic_name = 'topic_name'
col_num_docs = 'num_docs'
col_perc_docs = 'perc_docs'

col_doc_num = 'doc_num'
col_perc_topic_contrib = 'perc_topic_contrib'
col_topic_kw = 'topic_kw'
col_sentence = 'sentence'

col_ = ''

iques_1 = 'investor_and_expectation'
iques_2 = 'letter_to_shareholders'
iques_3 = 'customer_of_se'
iques_4 = 'se_competition'
iques_5 = 'se_business'
iques_6 = 'se_is_on_rise'
iques_7 = 'what_ques_to_ask'
iques_8 = 'se_is_on_decline'

idict_ques_mapper = {
    'Who would be the investors of a sales enablement business?  What would they expect? ': iques_1,
    'If sales enablement were to write a letter to shareholders describing how they performed and what they are going to do next year, what would it sound like?': iques_2,
    'Who is the customer of sales enablement? ': iques_3,
    "What is sales enablement's competition? ": iques_4,
    'If sales enablement were a business, what business would it be? ': iques_5,
    'If you think it is on the rise, please share why you think so?': iques_6,
    'Optional:  What question do you wish we would have asked? ': iques_7,
    'If you think it is in decline, please share why you think so?': iques_8
}

MANNUAL_STOP_WORDS = [' ', '"', ',', '.', '...', '/', '1', '5', ';', ':', '?', '\n', '\n\n', 'nodata'] + \
                     ['customer', 'expect', 'business', 'enablement', '-pron-', '-PRON-'] + \
                     ['company', 'team', 'people', 'pron', 'rep', 'function', 'enablement', 'se', 'think', 'na',
                      '-pron-', 'rep'] + \
                     ['year', 'organization']


# CUSTOM_STOP_WORDS = list( stop_words.ENGLISH_STOP_WORDS ) + MANNUAL_STOP_WORDS #+ idict_SW_mapper[col_main]


# idict_SW_mapper = dict.fromkeys( ilist_freeformQues_fmt, [] )
def get_sw_mapper():
    idict_SW_mapper = dict.fromkeys(list(idict_ques_mapper.values()), [])
    idict_SW_mapper[iques_1] = ['sale_enablement', 'sale', 'investor', 'pe', 'like', 'line', 'clear', 'key', 'level', 'include', 'invest',
                                'investment', 'ques1'] + [
                                   'value']  # , 'right', 'work', 'sell'] #+ ['expectation', 'value', 'tech', 'sell', 'end', 'op', 'real','service', 'suite', 'outcome']

    idict_SW_mapper[iques_2] = ['sale_enablement', 'salexxxx', '-PRON-'] + ['look', 'know', 'agree']

    idict_SW_mapper[iques_3] = ['sell', 'ultimately', 'selling', 'achieve', 'goal', 'level', 'work', 'prospect',
                                'serve', 'question', 'force', 'end', 'want', 'ques3', 'believe', 'internally',
                                'great'] + ['sale_sale', 'salexxxxxx    ', 'suite', 'define', 'line', 'sale_enablement', 'enablement', 'enable', 'ques33',
                                            'person', 'role', 'include'] + ['head', 'lead'] + ['leader', 'market', 'buy']

    idict_SW_mapper[iques_4] = ['competition', 'thing', 'org', 'sure', 'definition'] + ['know', 'sale', 'sale_enablement', 'way', 'align', 'entire']
    idict_SW_mapper[iques_5] = ['sale', 'change', 'create', 'sure', 'time', 'truly', 'end'] + ['sale_enablement', 'truly', 'believe']

    idict_SW_mapper[iques_6] = ['sale_enablement', 'sale', '-PRON-'] +  \
                               ['execute', 'point', 'use', 'field', 'issue', 'lot', 'training', 'quota', 'goal', 'role', 'system', 'key'] + \
                               ['need', 'tool', 'way', 'enable', 'define', 'game', 'happen', 'today', 'wear'] + ['increase', 'rise', 'new'] + ['growth']

    idict_SW_mapper[iques_7] = ['sale_enablement', 'question'] + ['love', 'interested', 'add']

    idict_SW_mapper[iques_8] = ['sale_enablement', 'decline']

    return idict_SW_mapper


iCUSTOM_WORD_LEMMA = {'consult': 'consultancy', 'consulting': 'consultancy', 'consultant': 'consultancy', 'consultation': 'consultancy'}


def get_bigram_common_terms_mapper():
    idict_bigram_common_terms_mapper = dict.fromkeys(list(idict_ques_mapper.values()), [])
    idict_bigram_common_terms_mapper[iques_1] = ['cro', 'cmo', 'coo', 'cfo', 'ceo', 'end']
    idict_bigram_common_terms_mapper[iques_3] = ['ceo', 'cfo', 'cro', 'cmo']
    idict_bigram_common_terms_mapper[iques_6] = ['-PRON-']

    return idict_bigram_common_terms_mapper


def get_topic_name_mapper():
    idict_topic_name_mapper = dict.fromkeys(list(idict_ques_mapper.values()), None)
    # noinspection PyTypeChecker
    idict_topic_name_mapper[iques_1] = {
        0: 'marketing_reps/venture_capitalists/outcomes_and_efficiency',
        1: 'high-qulatiy_growth_returns',
        2: 'top_company_leaders/productivity_revenue'
    }

    idict_topic_name_mapper[iques_3] = {
        0: 'customer&clinet_facing_roles/buyers',
        1: 'sale_team&manager/marketing',
        2: 'all_sales_dignitary'
        }

    idict_topic_name_mapper[iques_4] = {
        0: 'solution/initiatives/budget',
        1: 'sales/marketting',
        2: 'leadership_and_strategy',
        3: 'consultants/LnD',
        4: 'change_resistance/training/status_quo'
    }

    idict_topic_name_mapper[iques_5] = {
        0: 'innovative/creative/adding_value',
        1: 'strategic_critical_roles',
        2: 'professional_development/',
        3: 'service_business/help_business_growth',
        4: 'management/consultancy'
    }

    idict_topic_name_mapper[iques_6] = {
        0: 'marketing/more_revenue',
        1: 'new_se_jobs',
        2: 'sale_team_growth',
        3: 'incr_strategic_and_critical_role_of_se'
    }

    idict_topic_name_mapper[iques_7] = {
        0: "changes_in_se/challanges_in_se/se_managment",
        1: "how_to_sell_se/impact_on_se",
        2: "se_measurement/how_Se_looks_like"
    }

    return idict_topic_name_mapper


def get_hyperparam_mapper():
    idict_hyperparam_mapper = dict.fromkeys(list(idict_ques_mapper.values()), None)

    idict_hyperparam_mapper[iques_1] = {'bigrm_min_count': 3, 'num_topic': 3, 'topN_kw': 20, 'result': 'kw=482, bigram=4, score=0.49'}
    idict_hyperparam_mapper[iques_3] = {'bigrm_min_count': 4, 'num_topic': 3, 'topN_kw': 16, 'result': 'kw=220, bigram=5, score=0.58'}
    idict_hyperparam_mapper[iques_4] = {'bigrm_min_count': 2, 'num_topic': 5, 'topN_kw': 18, 'result': 'kw=323, bigram=8,  score=0.55'}
    idict_hyperparam_mapper[iques_5] = {'bigrm_min_count': 12, 'num_topic': 5, 'topN_kw': 18, 'result': 'kw=387, bigram=0,  score=0.47'}
    idict_hyperparam_mapper[iques_6] = {'bigrm_min_count': 4, 'num_topic': 4, 'topN_kw': 20, 'result': 'kw=475, bigram=2,  score=0.42'}
    idict_hyperparam_mapper[iques_7] = {'bigrm_min_count': 2, 'num_topic': 3, 'topN_kw': 12, 'result': 'kw=222, bigram=0,  score=0.45'}

    return idict_hyperparam_mapper


def get_dict_topN_kw_arr_mapper():
    idict_topN_kw_arr_mapper = dict.fromkeys(list(idict_ques_mapper.values()), {})
    idict_topN_kw_arr_mapper[iques_1] = {2: 20, 3: 20, 4: 18, 5: 18, 6:16, 7: 16, 8: 15}
    idict_topN_kw_arr_mapper[iques_3] = {2: 18, 3: 16, 4: 12, 5: 10, 6: 8}
    idict_topN_kw_arr_mapper[iques_6] = {2: 22, 3: 22, 4: 20, 5: 20, 6: 18, 7: 18, 8: 18}
    idict_topN_kw_arr_mapper[iques_7] = {2: 16, 3: 12, 4: 9, 5: 8, 6: 8, 7: 6}

    return idict_topN_kw_arr_mapper


def update_spacy_stopwords(col_main=None, col_main_lastUsed=None, sw_list=()):
    global nlp

    idict_SW_mapper = get_sw_mapper()
    new_stop_words = CUSTOM_STOP_WORDS + idict_SW_mapper[col_main] + sw_list
    # CUSTOM_STOP_WORDS = new_stop_words.copy()

    if col_main_lastUsed is not None:
        depr_stop_words = idict_SW_mapper[col_main_lastUsed]
        # print(depr_stop_words)
        for sw in depr_stop_words:
            if nlp.vocab[sw].is_stop is True:
                nlp.Defaults.stop_words.remove(sw)
                nlp.vocab[sw].is_stop = False

    for sw in new_stop_words:
        nlp.Defaults.stop_words.add(sw)
        nlp.vocab[sw].is_stop = True


def check_stopword(sw):
    return nlp.vocab[sw].is_stop


def merge_ques_main(data=None):
    free_from_ques = ['Who would be the investors of a sales enablement business?  What would they expect? ',
                      'If sales enablement were to write a letter to shareholders describing how they performed and what they are going to do next year, what would it sound like?',
                      'Who is the customer of sales enablement? ',
                      "What is sales enablement's competition? ",
                      'If sales enablement were a business, what business would it be? ',
                      'If you think it is on the rise, please share why you think so?',
                      'Optional:  What question do you wish we would have asked? ',
                      'If you think it is in decline, please share why you think so?']

    def merge_ques(cols):
        ls = list(cols.values)
        ls = ' # '.join(ls)
        return ls

    X = data.copy()
    X = X.fillna(NA_DATA)
    X[col_all_fques] = X[free_from_ques].apply(merge_ques, axis=1)

    return X


def print_ff_ques(dict_ques_mapper=None, list_freeformQues_fmt=None):
    print("Total Questions to be Analyzed: {}\n".format(len(list_freeformQues_fmt)))
    for ques, que_fmt in dict_ques_mapper.items():
        print("{:25s} : {}".format(que_fmt, ques))
        print("_" * 127)


# noinspection PyUnusedLocal
def print_ff_ques_dfFormat(dict_ques_mapper=None, list_freeformQues_fmt=None):
    print("Total Questions to be Analyzed: {}\n".format(len(list_freeformQues_fmt)))
    data = [list(idict_ques_mapper.values()), list(idict_ques_mapper.keys())]
    df = pd.DataFrame(data).T
    df.columns = ['ques_name', 'ques_description']

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.colheader_justify', 'light',
                           'display.width', 2000, 'display.max_colwidth', 500):
        df = df.stack().str.lstrip().unstack()
        df = df.style.set_properties(**{'text-align': 'left'})

    return df


def get_ff_question_info(data=None, dict_ques_mapper=None, nTop_ques=8, verbose=True):
    X = data.copy()

    ilist_questions = list(X.columns)
    ilist_questions = ilist_questions[2:]

    result = []
    for col in ilist_questions:
        nu = X[col].nunique()
        result.append([col, nu])

    X_nuAns = pd.DataFrame(data=result, columns=[col_ques, col_nuAns])
    X_nuAns[col_puAns] = X_nuAns[col_nuAns] / len(data)
    X_nuAns[col_puAns] = X_nuAns[col_puAns].round(2)
    X_nuAns = X_nuAns.sort_values(by=col_puAns, ascending=False).reset_index(drop=True)

    ilist_freeformQues = X_nuAns[col_ques][:nTop_ques]
    ilist_freeformQues_fmt = list(dict_ques_mapper.values())

    if verbose:
        print("_" * 50, "get_ff_question_info", "_" * 50)
        print("num total ques        :", len(ilist_questions))
        print("num freeform ques     :", len(ilist_freeformQues))
        print("num freeform_fmt ques :", len(ilist_freeformQues_fmt))
        print("len dict_ques_mapper  :", len(dict_ques_mapper))

    return ilist_questions, ilist_freeformQues, ilist_freeformQues_fmt


########################################## plot func #################################################################\

def extract_sent_score(d):
    neg = d['neg']
    pos = d['pos']

    if neg > pos:
        return -neg
    else:
        return pos


def get_sentiment_data(data=None, colList_meta=None):
    # get base raw data
    X = data.copy()
    X = X.fillna(NA_DATA)
    X = X.rename(columns=idict_ques_mapper)

    colList_ques = list(idict_ques_mapper.values()) + [col_all_fques]
    colList = colList_meta + colList_ques
    X = X[colList]

    # melt
    df = pd.melt(frame=X, id_vars=col_user, value_vars=colList_ques, var_name=col_ques, value_name=col_text)

    # get na data info
    df_na = df[df[col_text] == NA_DATA].reset_index(drop=True).copy()
    df_na = df_na[col_ques].value_counts().reset_index().rename(columns={col_ques: col_num_na, 'index': col_ques})
    df_na[col_perc_na] = np.round(df_na[col_num_na] / len(data), 2)

    # filter na data
    df = df[df[col_text] != NA_DATA].reset_index(drop=True)

    # get sentiment score
    # df[col_sent] = df[col_text].map(lambda text: TextBlob(text).sentiment.polarity)
    ### or
    obj_sid = SentimentIntensityAnalyzer()
    df[col_sent] = df[col_text].apply(lambda text: obj_sid.polarity_scores(text))

    df[col_sent] = df[col_sent].apply(lambda score_dict: score_dict['compound'])
    """
    df[col_sent] = df[col_sent].apply( lambda score_dict : extract_sent_score(score_dict) )
    """
    df[col_sent_type] = df[col_sent].apply(lambda score: 'pos' if score >= 0 else 'neg')

    # num words per response
    df[col_num_words] = df[col_text].apply(lambda text: len(text.split(' ')))
    X_sent = df.copy()

    # array to get sorted list of ques : by sent score"
    arr = X_sent.groupby(by=col_ques)[col_sent].mean().sort_values()[::-1]
    order_ques = list(arr.index.values)

    return X_sent, df_na, order_ques


def plot_sentiment_boxplot(data=None, order_ques=None, offset_path=None, im_path=None, save_plot=True, show_plot=True):
    plot_df = data.copy()

    plt.figure(figsize=(16, 7))
    sns.set_style("darkgrid")
    sns.boxplot(x=col_ques, y=col_sent, data=plot_df, order=order_ques, palette=ilist_color_0)

    main_label = 'Sentiment of User Response\n#users : {}'.format(plot_df[col_user].nunique())
    xlabel = 'question'
    ylabel = 'sentiment score'
    plt.title(label=main_label, fontsize=20, color='blue')
    plt.xlabel(xlabel=xlabel, fontsize=16, color='red')
    plt.ylabel(ylabel=ylabel, fontsize=16, color='red')
    plt.tick_params(axis='x', labelsize=15, pad=5, rotation=25)
    plt.ylim(-1.05, +1.05)

    if save_plot:
        image_name = '01_02_sent_box_v1.png'
        offset_path = offset_path
        PATH = im_path + offset_path + image_name
        plt.tight_layout()
        plt.savefig(PATH)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_sentiment_tree(data=None, offset_path=None, im_path=None, save_plot=True, show_plot=True, save_html=True):
    plot_df = data.copy()
    plot_df = plot_df[plot_df[col_ques] != col_all_fques].reset_index(drop=True)

    col_prim = col_ques
    col_sec = col_sent_type

    col_size = col_num_words
    col_color = col_sent

    plot_df['base'] = 'User Response'
    title_name = "User Response length and Sentiment Insights" + '<br>' + "size denotes num_words"

    fig = px.treemap(plot_df, path=['base', col_prim, col_sec], values=col_size,
                     color=col_color, hover_data=[col_ques, col_num_words],
                     color_continuous_scale='RdBu', title=title_name,
                     color_continuous_midpoint=0)
    # , labels={col_perc_saving:'% saving'}

    fig.update_layout(
        title={
            'text': title_name,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="size denotes num_words",
        width=1000,
        height=750)

    if save_plot:
        image_name = '01_03_nwSent_tree_v1'
        offset_path = offset_path
        PATH = im_path + offset_path + image_name
        fig.write_image("{}.png".format(PATH), scale=2.5)

        if save_html:
            fig.write_html("{}.html".format(PATH))

    if show_plot:
        fig.show()
    else:
        plt.close()


def plot_sentiment_sunburst(data=None, offset_path=None, im_path=None, save_plot=True, show_plot=True, save_html=True):
    plot_df = data.copy()
    plot_df = plot_df[plot_df[col_ques] != col_all_fques].reset_index(drop=True)

    col_prim = col_ques
    col_sec = col_sent_type

    col_size = col_num_words
    col_color = col_sent

    plot_df['base'] = 'User Response'
    title_name = "User Response length and Sentiment Insights" + '<br>' + "size denotes num_words"

    fig = px.sunburst(plot_df, path=['base', col_prim, col_sec], values=col_size,
                      color=col_color, hover_data=[col_ques, col_num_words],
                      color_continuous_scale='RdBu', title=title_name,
                      color_continuous_midpoint=0)

    """
    fig = px.treemap(plot_df, path=['base', col_prim, col_sec], values=col_size,
                      color=col_color, hover_data=[col_ques, col_num_words],
                      color_continuous_scale='RdBu', title=title_name,
                      color_continuous_midpoint=0 )
    # , labels={col_perc_saving:'% saving'}
    """

    fig.update_layout(
        title={
            'text': title_name,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="size denotes num_words",
        width=1000,
        height=750)

    if save_plot:
        image_name = '01_04_nwSent_sun_v1'
        offset_path = offset_path
        PATH = im_path + offset_path + image_name
        fig.write_image("{}.png".format(PATH), scale=2.5)

        if save_html:
            fig.write_html("{}.html".format(PATH))

    if show_plot:
        fig.show()
    else:
        plt.close()


def plot_numWords_hist(data=None, offset_path=None, im_path=None, save_plot=True, show_plot=True):
    plot_df = data.copy()
    plot_df = plot_df[plot_df[col_ques] != col_all_fques].copy()

    arr = plot_df.groupby(by=col_ques)[col_num_words].mean().sort_values()[::-1]
    order_ques = list(arr.index.values)

    sns.set_style('darkgrid')

    # colList = [i+'_nw' for i in ilist_freeformQues_fmt]
    colList = [i + '_nw' for i in order_ques]

    nplot = len(colList)
    nc = 2
    nr = int(np.floor(nplot / nc))

    fig, axis = plt.subplots(nr, nc, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.1)

    xlabel = 'num words per answers'
    ylabel = '# answers'
    main_label = 'Distribution of num words per Answer'

    ind = 0
    for i_main in range(nr):
        for j_main in range(nc):
            ax = axis[i_main, j_main]

            col = colList[ind]
            df = plot_df[plot_df[col_ques] == col[:-3]].copy()
            df = df[df[col_num_words] <= 120]

            arr = df[col_num_words].values
            # print(plot_df[col_ques])
            # print(col)

            ax.hist(arr, bins=20)
            ind += 1

            ax.set_title(label=col[:-3], fontsize=14, color='red')
            ax.set_xlabel(xlabel=xlabel, fontsize=12, color='black')
            ax.set_ylabel(ylabel=ylabel, fontsize=12, color='black')
            # plt.tick_params(axis='both', labelsize=25, pad=5)
            # ax.set_xlim(0,120)

    fig.text(0.5, 0.95, main_label, ha='center', va='center', rotation='horizontal', size=20, color='blue')

    if save_plot:
        image_name = '01_01_nw_dist_v1.png'
        offset_path = offset_path
        PATH = im_path + offset_path + image_name
        plt.tight_layout()
        plt.savefig(PATH)

    if show_plot:
        plt.show()
    else:
        plt.close()


# noinspection PyUnusedLocal
def make_word_cloud(obj_vectorizer=None, data_tfidf=None, col_main=None, only_neg_sent=False, X_sent=None,
                    figsize=(16, 10),
                    offset_path=None, im_path=None, save_plot=True, show_plot=True):
    obj_vectorizer = obj_vectorizer
    data_tfidf = data_tfidf

    sns.set_style('white')
    plt.figure(figsize=figsize)

    # check calc
    freqs = [(word, data_tfidf.getcol(idx).sum()) for word, idx in obj_vectorizer.vocabulary_.items()]
    w = WordCloud(width=800, height=600, mode='RGBA', background_color='white', max_words=2000).fit_words(dict(freqs))
    plt.imshow(w)
    # plt.axis(&quot;off&quot;)

    plt.xticks([])
    plt.yticks([])
    main_label = col_main
    plt.title(label=main_label, fontsize=20, color='blue')

    # image_name = None
    if save_plot:
        if only_neg_sent:
            image_name = '07_00_kw_wordcloud_neg_v1.png'
        else:
            image_name = '07_00_kw_wordcloud_v1.png'
        image_name = '07_{}_{}'.format(col_main, image_name)
        PATH = im_path + offset_path + image_name
        plt.tight_layout()
        plt.savefig(PATH)

    if show_plot:
        plt.show()
    else:
        plt.close()


def make_word_cloud_wrapper(col_main=None, data=None, only_neg_sent=False, X_sent=None, offset_path=None, im_path=None,
                            save_plot=True, show_plot=True):
    X = data.copy()
    if only_neg_sent:
        X = pd.merge(left=data, right=X_sent, on=col_user)
        X = X[X[col_sent_type] == 'neg']
        X = X[X[col_ques] == col_main]
        X = X.reset_index(drop=True)

    print(X.shape)
    wc_obj_tfidf_vectorizer, wc_tfidf_matrix = get_tfidf_data(col_main=col_main, data=X)

    make_word_cloud(obj_vectorizer=wc_obj_tfidf_vectorizer, data_tfidf=wc_tfidf_matrix, col_main=col_main,
                    only_neg_sent=only_neg_sent, X_sent=X_sent,
                    offset_path=offset_path, im_path=im_path, save_plot=save_plot, show_plot=show_plot)


############################################################################################################################


def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-Z ]'  # remove every thing except chars and space
    text = re.sub(pat, ' ', text)

    pat = ' +'  # 1 or more spaces
    text = re.sub(pat, ' ', text)

    text = text.strip()
    return text


# call function
# text = "07 Not sure@ if this % was #fun! 558923 What do# you think** of it.? $500USD! ! ... .... ....... .."
# remove_special_characters(text)


def get_tfidf_data(col_main=None, data=None):
    # col = col_main
    # col_token = col + '_token'

    X = data.copy()
    X = X.fillna(NA_DATA)
    X = X.rename(columns=idict_ques_mapper)

    # colList = icolList_meta+[col_main]
    # X = X[colList]

    obj_tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                           min_df=2, stop_words='english',
                                           # use_idf=True, tokenizer=spacy_tokenizer_lemmatizer)#, ngram_range=(1,2))
                                           use_idf=True, tokenizer=spacy_tokenizer_lemmatizer, ngram_range=(1, 2))

    # %time
    tfidf_matrix = obj_tfidf_vectorizer.fit_transform(X[col_main])  # fit the vectorizer to synopses
    return obj_tfidf_vectorizer, tfidf_matrix


def get_tfidf_data_adv(col_nlp=None, data=None, max_df=0.95, min_df=2, stop_words_tfidf=None):
    """
    data : ml data, clean, tokenized
    """
    X = data.copy()

    def dummy_fun(doc):
        return doc

    obj_tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words_tfidf,
                                           # use_idf=True, tokenizer=spacy_tokenizer_lemmatizer)#, ngram_range=(1,2))
                                           use_idf=True, tokenizer=dummy_fun, lowercase=False, ngram_range=(1, 1))

    # %time
    tfidf_matrix = obj_tfidf_vectorizer.fit_transform(X[col_nlp].tolist())  # fit the vectorizer to synopses

    return obj_tfidf_vectorizer, tfidf_matrix


def print_tfidf_info(tfidf_matrix=None, obj_tfidf_vectorizer=None, verbose=True):
    if verbose:
        print("_" * 50, "print_tfidf_info", "_" * 50)
        print("shape of tfidf_matrix    : {}".format(str(tfidf_matrix.shape)))

        terms = obj_tfidf_vectorizer.get_feature_names()
        # val = [i for i in terms if len(i.split()) >= 2]
        print("num text in tfidf_matrix : {}".format(len(terms)))


def spacy_tokenizer_lemmatizer(text):
    """
    1. remove spec char
    1. tokenize
    2. lowercase
    3. remove stop words
    4. lemmatization
    """

    ls_text = text.split(' ')
    ls_text = [remove_special_characters(i) for i in ls_text]
    ls_text = [i.strip() for i in ls_text]
    ls_text = [i for i in ls_text if i not in CUSTOM_STOP_WORDS]
    text = ' '.join(ls_text)

    doc = nlp(text)

    # tokenizer = nlp.Defaults.create_tokenizer(nlp)
    # tokens = tokenizer(text)

    lemma_list = []
    for token in doc:
        if token.is_stop is False:
            token_text = token.lemma_.lower()

            if nlp.vocab[token_text].is_stop is False:
                # token_text = remove_special_characters(token_text)
                # token_text = token_text.strip()

                if len(set(token_text)) >= 2:
                    lemma_list.append(token_text)

    # lemma_list = [i for i in lemma_list if i not in CUSTOM_STOP_WORDS]

    return lemma_list


################################################## Topic midelling Start ##############################################################


def reformat_bigram(token_list):
    for i_main, list_word in enumerate(token_list):
        for j_main, word in enumerate(list_word):
            new_word = '************'.join(word.split(' '))
            token_list[i_main][j_main] = new_word

    return token_list


def get_token_data(data=None, col_main=None, verbose=True):
    X = data.copy()
    X = X.fillna(NA_DATA)
    X = X.rename(columns=idict_ques_mapper)

    X[col_main + '_token'] = X[col_main].apply(spacy_tokenizer_lemmatizer)

    # docks_token = [ ' '.join(ls) for ls in X[col_main+'_token'].values ]
    docs = [i for i in X[col_main].values]
    # docks
    docs_token = list(X[col_main + '_token'].values)

    # len(docks_token)
    if verbose:
        print("_" * 50, "get_token_data", "_" * 50)
        print("Num docs         :", len(docs_token))
        print("Num unique words :", len(set([i for j in docs_token for i in j])))

    return docs, docs_token


def gensim_vectorizer(docs_token=None, verbose=True):
    # Create Dictionary
    id2word_dict = corpora.Dictionary(docs_token)
    id2word_dict.filter_extremes(no_below=2, no_above=0.90)

    # Create Corpus
    # texts = docks_token

    # Term Document Frequency
    corpus = [id2word_dict.doc2bow(sentence) for sentence in docs_token]

    if verbose:
        print("_" * 50, "gensim_vectorizer", "_" * 50)
        print("num docs in corpus       :", len(corpus))
        print("length of id2word_dict   :", len(id2word_dict))
        print("1st item of corpus       :", corpus[:1])
        # Human readable format of corpus (term-frequency)
        corpus_text_values = [[(id2word_dict[i], freq) for i, freq in cp] for cp in corpus[:1]]
        print("1st item of corpus, text :", corpus_text_values)

    return id2word_dict, corpus


def compute_coherence_values(id2word_dict=None, corpus=None, texts=None,
                             limit=5, start=2, step=3, chunksize=20, passes=20, verbose=True):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    id2word_dict : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """

    if verbose:
        print("_" * 45, "compute_coherence_values", "_" * 45)

    # coherence_values = []
    # model_list = []
    i_main = 0
    dict_result = {}

    for num_topics in range(start, limit, step):
        print(i_main, end=', ')
        i_main += 1

        # model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word_dict,
                                                num_topics=num_topics,
                                                random_state=SEED,
                                                update_every=1,
                                                chunksize=chunksize,
                                                passes=passes,
                                                alpha='auto',
                                                per_word_topics=True)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word_dict, coherence='c_v')
        coherence_score = coherencemodel.get_coherence()

        # model_list.append(model)
        # coherence_values.append( coherence_score )

        dict_result[num_topics] = [model, coherence_score]

    return dict_result


def find_best_genism_lda_model(id2word_dict=None, corpus=None, texts=None,
                               limit=5, start=2, step=3, chunksize=20, passes=20, topN_kw=20, topN_kw_arr=None,
                               verbose=True):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    id2word_dict : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    topN_kw_arr : dict { (topic_num:topN_kw) }
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """

    if verbose:
        print("_" * 45, "find_best_genism_lda_model", "_" * 45)

    # coherence_values = []
    # model_list = []
    i_main = 0
    dict_result = {}

    if topN_kw_arr is not None:
        #num_topics_arr = [i[0] for i in topN_kw_arr]
        num_topics_arr = list(topN_kw_arr.keys())
        num_topics_arr = sorted(num_topics_arr)
    else:
        num_topics_arr = [i for i in range(start, limit, step)]

    for num_topics in num_topics_arr:
        if topN_kw_arr is not None:
            topN_kw = topN_kw_arr[num_topics]

        print("ind: {}, num_topic:{}, topN_kw; {}".format(i_main, num_topics, topN_kw))
        #print(i_main, end=', ')
        i_main += 1

        model = train_gensim_lda_model(num_topics=num_topics, id2word_dict=id2word_dict, corpus=corpus, texts=texts,
                                       chunksize=chunksize, passes=passes, verbose=verbose)

        coherence_score = compute_coherence_score(model=model, texts=texts, id2word_dict=id2word_dict, topN_kw=topN_kw)

        dict_result[num_topics] = [model, coherence_score]

    return dict_result


def train_gensim_lda_model(modelling_type=None, num_topics=2, id2word_dict=None, corpus=None, texts=None,
                           chunksize=20, passes=20, topN_kw=20, verbose=True):
    if verbose and modelling_type == 'single':
        print("_" * 45, "train_gensim_lda_model", "_" * 45)
        print("num_topics          : {}".format(num_topics))
        print("len(id2word_dict)   : {}".format(len(id2word_dict)))
        print("len(corpus)         : {}".format(len(corpus)))
        print("len(texts)          : {}".format(len(texts)))
        print("chunksize           : {}".format(chunksize))
        print("passes              : {}".format(passes))
        print("topN_kw(for coh_sc) : {}".format(topN_kw))
        print()

    model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word_dict,
                                            num_topics=num_topics,
                                            random_state=SEED,
                                            update_every=1,
                                            chunksize=20,
                                            passes=20,
                                            alpha='auto',
                                            per_word_topics=True)
    return model


# parameterize topn
def compute_coherence_score(model=None, texts=None, id2word_dict=None, topN_kw=20):
    """
    tunanle param : topn ( based on which coh_score is calc )
    """
    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word_dict, coherence='c_v', topn=topN_kw)
    coherence_score = coherencemodel.get_coherence()
    return coherence_score


def train_single_genism_lda_model(modelling_type=None, num_topics=2, id2word_dict=None, corpus=None, texts=None,
                                  chunksize=20, passes=20, verbose=True, topN_kw=20, topN_kw_arr=None):
    if topN_kw_arr is not None:
        topN_kw = topN_kw_arr[num_topics]

    model = train_gensim_lda_model(modelling_type=modelling_type, num_topics=num_topics, id2word_dict=id2word_dict,
                                   corpus=corpus, texts=texts,
                                   chunksize=chunksize, passes=passes, topN_kw=topN_kw, verbose=verbose)
    coherence_score = compute_coherence_score(model=model, texts=texts, id2word_dict=id2word_dict, topN_kw=topN_kw)

    return model, num_topics, coherence_score


# noinspection PyUnusedLocal
def print_modelling_info(model=None, num_topics=None, coherence_score=None, topN_kw=20, topN_kw_arr=None, verbose=True):

    if topN_kw_arr is not None:
        topN_kw = topN_kw_arr[num_topics]

    # if(verbose and modelling_type == 'single'):
    if verbose:
        print("_" * 45, "Modelling completed", "_" * 45)
        # print("Modelling completed :")
        print("num topics      : {}".format(num_topics))
        print("coherence score : {} (topN_kw={})".format(round(coherence_score, 2), topN_kw))


def model_training_wrapper(num_topics=2, modelling_type='single',
                           id2word_dict=None, corpus=None, texts=None,
                           start=2, limit=4, step=1, chunksize=20, passes=20, topN_kw=20, topN_kw_arr=None,
                           verbose=True, offset_path=None, im_path=None, save_plot=True, show_plot=True):
    optimal_model = None
    optimum_num_cluster = None
    max_c_score = None

    if modelling_type == 'single':
        # only train single model
        optimal_model, optimum_num_cluster, max_c_score = train_single_genism_lda_model(modelling_type=modelling_type,
                                                                                        num_topics=num_topics,
                                                                                        id2word_dict=id2word_dict,
                                                                                        corpus=corpus, texts=texts,
                                                                                        chunksize=chunksize,
                                                                                        passes=passes, topN_kw=topN_kw, topN_kw_arr=topN_kw_arr,
                                                                                        verbose=verbose)

        print_modelling_info(model=optimal_model, num_topics=optimum_num_cluster, coherence_score=max_c_score, topN_kw=topN_kw, topN_kw_arr=topN_kw_arr,
                             verbose=True)

    elif modelling_type == 'multiple':
        # train multiple models
        # Can take a long time to run.

        dict_result = find_best_genism_lda_model(id2word_dict=id2word_dict, corpus=corpus, texts=texts,
                                                 start=start, limit=limit, step=step,
                                                 chunksize=chunksize, passes=passes, topN_kw=topN_kw, topN_kw_arr=topN_kw_arr,
                                                 verbose=verbose)

        optimal_model, optimum_num_cluster, max_c_score = coherence_score_plot(dict_result=dict_result, step=step,
                                                                               verbose=verbose, offset_path=offset_path,
                                                                               im_path=im_path, save_plot=save_plot,
                                                                               show_plot=show_plot)
        topN_kw = topN_kw_arr[optimum_num_cluster]
        print_modelling_info(model=optimal_model, num_topics=optimum_num_cluster, coherence_score=max_c_score,
                             topN_kw=topN_kw,
                             verbose=True)

    elif modelling_type == 'load_existing':
        print("Not implemented yet")

    else:
        print("Invalid modelling_type....!")

    return optimal_model, optimum_num_cluster, max_c_score


############################################## NLP model end ######################################################################


######################################### NLP model EDA start ######################################################################

def coherence_score_plot(dict_result=None, step=1, verbose=True, offset_path=None, im_path=None, save_plot=True,
                         show_plot=True):
    num_topic = list(dict_result.keys())

    # Show graph
    x = range(min(num_topic), max(num_topic) + 1, step)
    ls = list(dict_result.values())
    y = [i[1] for i in ls]

    # print(x)
    # print(y)
    max_score = max(y)
    max_score_ind = y.index(max_score)
    optimum_num_cluster = x[max_score_ind]
    optimal_model = dict_result[optimum_num_cluster][0]

    if verbose:
        print("best score             :", round(max_score, 2))
        print("optimum num of cluster :", optimum_num_cluster)

    plt.figure(figsize=(12, 7))
    sns.set_style("darkgrid")

    # plt.plot(x, y, '--b', linewidth=2.5, marker='o', markersize=15)
    plt.plot(x, y, '--', color='#303F9F', linewidth=3, marker='o', markersize=10, label='coherence score')

    plt.xlabel("num topics", color='black', fontsize=15)
    plt.ylabel("coherence score", color='black', fontsize=15)
    plt.legend("coherence_values", loc='best')

    plt.tick_params(axis='x', labelsize=15, pad=5)
    plt.axvline(x=optimum_num_cluster, ymin=0.0, ymax=1, linewidth=1, color='black', linestyle='--')

    if save_plot:
        image_name = '07_01_coherenceScore_line_v1.png'
        offset_path = offset_path
        PATH = im_path + offset_path + image_name
        plt.tight_layout()
        plt.savefig(PATH)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return optimal_model, optimum_num_cluster, round(max_score, 2)


def print_gensim_lda_topics(model=None, num_words=5, verbose=True):
    if verbose:
        print("=" * 50, "print_gensim_lda_topics", "=" * 50)
        print()

        topics = model.print_topics(num_words=num_words)
        for i, t in enumerate(topics):
            print("topic-", i)
            print(t)
            print('_' * 125)
            print()


def show_pyLDAvis(model=None, corpus=None, id2word_dict=None):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(model, corpus, id2word_dict)
    return vis


# import plotly.graph_objs as go
# import plotly.offline as py
def plot_ldaModel_difference_plotly(mdiff, annotation=None, col_main=None, use_topic_name=False, dict_topic_mapper=None,
                                    graph_width=600, graph_height=600,
                                    offset_path=None, im_path=None, save_plot=True, show_plot=True, save_html=True):
    """
    make it plotly express compatible
    """
    annotation_html = None
    if annotation is not None:
        annotation_html = [
            [
                "+++ {}<br>--- {}".format(", ".join(int_tokens), ", ".join(diff_tokens))
                for (int_tokens, diff_tokens) in row
            ]
            for row in annotation
        ]
    title_name = "Topic difference [jaccard distance]" + '<br>' + "ques : {}".format(col_main)

    num_topics = len(mdiff)
    topic_name_list = []
    # topic_name = None
    for topic_id in range(num_topics):
        if use_topic_name:
            topic_name = dict_topic_mapper[topic_id]
        else:
            topic_name = "Topic-" + str(topic_id)
        topic_name_list.append(topic_name)

    tickvals = [i for i in range(num_topics)]
    ticktext = topic_name_list

    data = go.Heatmap(z=mdiff, colorscale='RdBu', text=annotation_html, name="Topic Diff", showlegend=True)
    layout = go.Layout(
        width=graph_width, height=graph_height,
        title={
            'text': title_name,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis=dict(title="topic", tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(title="topic", tickvals=tickvals, ticktext=ticktext),
    )
    fig = go.Figure(dict(data=[data], layout=layout))

    if save_plot:
        image_name = '07_02_topicDiff_heat_v1'
        image_name = '07_{}_{}'.format(col_main, image_name)
        offset_path = offset_path
        PATH = im_path + offset_path + image_name
        fig.write_image("{}.png".format(PATH), scale=2.5)

        if save_html:
            fig.write_html("{}.html".format(PATH))

    if show_plot:
        fig.show()
    else:
        plt.close()


# Finding the dominant topic in each sentence
def format_topics_sentences(ldamodel=None, corpus=None, texts=None, data_ml=None, dict_topic_mapper=None, topN_kw=5,
                            verbose=True):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):

        # new change
        row = row[0]

        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num, topn=topN_kw)
                # print( len(wp) )
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    # Format
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    df_dominant_topic['Dominant_Topic'] = df_dominant_topic['Dominant_Topic'].astype('int')

    # adding user ids to the data
    if data_ml is not None:
        df = data_ml[[col_user]].copy()
        df_dominant_topic = pd.concat(objs=[df, df_dominant_topic], axis=1)

    if dict_topic_mapper is not None:
        df_dominant_topic[col_topic_name] = df_dominant_topic['Dominant_Topic'].copy()
        df_dominant_topic[col_topic_name] = df_dominant_topic[col_topic_name].replace(dict_topic_mapper)
    else:
        df_dominant_topic[col_topic_name] = df_dominant_topic['Dominant_Topic'].copy()
        df_dominant_topic[col_topic_name] = df_dominant_topic[col_topic_name].apply(
            lambda topic_id: "topic-{}".format(topic_id))

    if verbose:
        print("_" * 50, "format_topics_sentences", "_" * 50)
        print("df_dominant_topic shape :", df_dominant_topic.shape)

    return df_dominant_topic


# Find the most representative document for each topic
# Best way to do sanity check
def get_topN_topics(topic_df=None, topN=1, verbose=True):
    top_topics = pd.DataFrame()
    gp_obj = topic_df.groupby("Dominant_Topic")

    for i, gp in gp_obj:
        top = gp.sort_values(['Topic_Perc_Contrib'], ascending=False).head(topN)
        top_topics = pd.concat(objs=[top_topics, top], axis=0)

    if verbose:
        print("_" * 50, "get_topN_topics", "_" * 50)
        print("total records       :", len(top_topics))
        print("num topics selected :", topN)
        print("_" * 128)
        print()

    return top_topics.reset_index(drop=True)


def nice_print_doc_topics(data=None, col_main=None):
    X = data.copy()
    X = X.sort_values(['Dominant_Topic', 'Topic_Perc_Contrib'], ascending=False).reset_index(drop=True)

    print("Ques :", col_main)

    for row in X.iterrows():
        topic_num = row[1]['Dominant_Topic']
        perc_contrib = row[1]['Topic_Perc_Contrib']
        kw = row[1]['Keywords']
        text = row[1]['Text']

        matches = []
        # print(type(kw))
        # break
        for word in kw.split(', '):
            if ',' in word:
                print(word)

            for main_word in text.split():
                if word.lower() in main_word.lower().strip():
                    matches.append(word)
                    break
        matches = list(set(matches))
        # print(len(kw.split()))

        print("Topic (Contribution) : {} ({})".format(topic_num, perc_contrib))
        print("Top keywords         : {}".format(kw))
        print("Matched kws          : {}".format(matches))
        print("Original text        :", text)
        # print(text)
        print("_" * 125)
        print()
    return


# Topic distribution across documents
# Finally, we want to understand the VOLUME and DISTRIBUTION of topics in order to judge how widely it was discussed.
# The below table exposes that information
# add avg cvg
def dominant_topic_stats(data=None, use_topic_name=False, verbose=True):
    X = data.copy()
    col_topic_ref = 'Dominant_Topic'
    if use_topic_name:
        col_topic_ref = col_topic_name

    X = X[col_topic_ref].value_counts().reset_index().rename(
        columns={'index': col_topic_num, col_topic_ref: col_num_docs})
    X[col_perc_docs] = np.round(X[col_num_docs] / X[col_num_docs].sum(), 2)

    df = data.groupby(col_topic_ref)['Topic_Perc_Contrib'].mean().reset_index(). \
        rename(columns={'Topic_Perc_Contrib': 'avg_contrib', col_topic_ref: col_topic_num})
    df['avg_contrib'] = df['avg_contrib'].round(2)

    X = pd.merge(left=X, right=df, on=col_topic_num)

    if verbose:
        print("print some stats")

    return X


def plot_percUserPerTopic(data=None, col_main=None, graph_width=1000, graph_height=450, limit_y_axis=None,
                          offset_path=None, im_path=None, save_plot=True, show_plot=True, save_html=True):
    plot_df = data.copy()

    arr_x = list(plot_df[col_topic_num].values)
    arr_x = [str(i) for i in arr_x]

    arr_y = plot_df[col_perc_docs].values

    fig = px.bar(x=arr_x, y=arr_y)

    title_name = "% User per Topic" + '<br>' + "ques : {}".format(col_main)
    fig.update_layout(
        title={
            'text': title_name,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Topic",
        yaxis_title="% user",
        width=graph_width,
        height=graph_height)

    if limit_y_axis is not None:
        fig.update_layout(
            yaxis=dict(range=limit_y_axis)
        )

    if save_plot:
        image_name = '07_05_percUserPerTopic_bar_v1'
        image_name = '07_{}_{}'.format(col_main, image_name)
        offset_path = offset_path
        PATH = im_path + offset_path + image_name
        fig.write_image("{}.png".format(PATH), scale=2.5)

        if save_html:
            fig.write_html("{}.html".format(PATH))

    if show_plot:
        fig.show()
    else:
        plt.close()


def plot_spacy_topic_kws_simple(model=None, id2word_dict=None, col_main=None, topN=4,
                                use_topic_name=False, dict_topic_mapper=None,
                                graph_width=1000, graph_height=550,
                                offset_path=None, im_path=None, save_plot=True, show_plot=True, save_html=True):
    num_topics = model.num_topics

    subplot_title_list = []
    for topic_id in range(num_topics):
        if use_topic_name:
            topic_name = dict_topic_mapper[topic_id]
        else:
            topic_name = "Topic-" + str(topic_id)
        subplot_title_list.append(topic_name)

    fig = make_subplots(rows=1, cols=num_topics, subplot_titles=subplot_title_list)
    # topic_name = ''
    for topic_id in range(num_topics):
        top_words = model.get_topic_terms(topicid=topic_id, topn=topN)
        arr_wd = [id2word_dict[wid] for wid, wt in top_words]
        arr_wt = [wt for wid, wt in top_words]
        # arr_ind = [i for i in range(len(arr_wd))]

        topic_name = subplot_title_list[topic_id]

        fig.add_trace(go.Bar(x=arr_wt[::-1], y=arr_wd[::-1], orientation='h', name=topic_name, ), row=1,
                      col=topic_id + 1)

    title_name = "Top Keywords per Topic" + '<br>' + "ques : {}".format(col_main)
    fig.update_layout(
        title={
            'text': title_name,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="keyword weight",
        # annotations=[{"text": "First Subplot"}, {"text": "2 Subplot"}, {"text": "3 Subplot"}],
        width=graph_width,
        height=graph_height)

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12, color='black')

    if save_plot:
        image_name = '07_04_topicKws_bar_v1'
        image_name = '07_{}_{}'.format(col_main, image_name)
        offset_path = offset_path
        PATH = im_path + offset_path + image_name
        fig.write_image("{}.png".format(PATH), scale=2.5)

        if save_html:
            fig.write_html("{}.html".format(PATH))

    if show_plot:
        fig.show()
    else:
        plt.close()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def cstr(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)


def nice_print_doc_topics_colored(data=None, col_main=None, use_topic_name=False, sentence_chunk_size=10,
                                  num_kw_print=5):
    X = data.copy()
    dict_rename = {'Dominant_Topic': col_topic_num, 'Document_No': col_doc_num,
                   'Topic_Perc_Contrib': col_perc_topic_contrib, 'Keywords': col_topic_kw, 'Text': col_sentence}
    X = X.rename(columns=dict_rename)

    topic_ls = list(X[col_topic_num].unique())
    # num_topics = X[col_topic_num].nunique()

    print("\nQues : {}\n".format(col_main))

    for i_main, topic_num in enumerate(topic_ls):
        df = X[X[col_topic_num] == topic_num].reset_index(drop=True).copy()
        kw = df.loc[0, col_topic_kw]
        kw_ls = kw.split(', ')

        color = ilist_color_1[i_main]

        # print("="*60, "Topic-"+str(topic_num), "="*60)
        if not use_topic_name:
            print("Topic-" + str(topic_num))
        else:
            topic_name = df.loc[0, col_topic_name]
            print("Topic-{} : {}".format(topic_num, topic_name))
        print(kw_ls[:num_kw_print])

        for row in df.iterrows():
            # perc_topic_contrib = row[1][col_perc_topic_contrib]

            sentence = row[1][col_sentence]
            # sentence_filtered = remove_special_characters(sentence)
            sentence_ls = sentence.split(" ")
            sentence_ls = [i.lower().strip() for i in sentence_ls]

            # sentence_ls_matchInd = []
            sentence_ls_colored = []
            # for each word in sentence
            for ind, word in enumerate(sentence_ls):
                # for each kw in topic
                for kw in kw_ls:
                    """
                    if kw in word:
                        word = cstr(word, color=color)
                    """
                    kw_split_ls = kw.split('_')
                    for kw_split in kw_split_ls:
                        if kw_split in word:
                            word = cstr(word, color=color)

                sentence_ls_colored.append(word)

            # sentence_colored = ' '.join( sentence_ls_colored )
            # sentence_colored = cstr( sentence_colored, color='black')
            sentence_colored_chunked = list(chunks(sentence_ls_colored, sentence_chunk_size))
            for sentence_colored in sentence_colored_chunked:
                sentence_colored = ' '.join(sentence_colored)
                sentence_colored = cstr(sentence_colored, color='black')
                display(html_print(sentence_colored))

            print("_" * 130)

            # break
        # break


def nlp_remove_punctuation(X=None, col_nlp='col_nlp', punc_re=None):
    # Remove punctuation

    if punc_re is None:
        # punc_re = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
        punc_re = r'[^a-zA-Z :@-]'  # remove every thing except chars and space
        ###punc_re =  r'[^a-zA-Z ]' # remove every thing except chars and space
        # X[col_main] = X[col_main].map(lambda x: re.sub('[,/.!?]', ' ', x) )

    X[col_nlp] = X[col_nlp].map(lambda x: re.sub(punc_re, ' ', x))

    # covert ' 'xn to ' '1
    X[col_nlp] = X[col_nlp].map(lambda x: re.sub(' +', ' ', x))

    return X


def nlp_to_lowercase(X=None, col_nlp='col_nlp'):
    # Convert the titles to lowercase
    X[col_nlp] = X[col_nlp].map(lambda x: x.lower())
    return X


def sent_to_words(X=None, deacc=True, col_nlp='col_clean'):
    # deacc=True removes punctuations
    # lambda sent_to_words_converted = gensim.utils.simple_preprocess(sentence, deacc=deacc)
    X[col_nlp] = X[col_nlp].map(lambda sentence: gensim.utils.simple_preprocess(sentence, deacc=deacc))

    return X


def nlp_make_bigram(X=None, col_nlp='col_nlp', min_count=1, threshold=0.1, common_terms=()):
    """
    It needs X[col_nlp] as ['aa', 'bb', 'cc', 'dd']
    
    The two important arguments to Phrases are min_count and threshold.
    The higher the values of these param, the harder it is for words to be combined.
    """
    # bigram and trigram models
    # higher threshold fewer phrases.
    list_sentences = X[col_nlp].tolist()
    model_bigram = gensim.models.Phrases(sentences=list_sentences, min_count=min_count, threshold=threshold,
                                         common_terms=common_terms)

    X[col_nlp] = X[col_nlp].map(lambda sentence: model_bigram[sentence])

    # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    return X


def nlp_filter_stopwords(X=None, col_nlp='col_nlp', use_nltk=True, use_spacy=True, custom_sw=(), only_return_sw=False):
    """
    Need, sentence=['aa', 'bb']
    """
    nltk_sw_ls = []
    spacy_sw_ls = []

    if use_nltk:
        nltk_sw_ls = list(nltl_sw.words('english'))
    if use_spacy:
        local_nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        spacy_sw_ls = list(local_nlp.Defaults.stop_words)

    all_sw_ls = nltk_sw_ls + spacy_sw_ls + custom_sw
    all_sw_ls = list(set(all_sw_ls))

    if only_return_sw:
        return all_sw_ls

    X[col_nlp] = X[col_nlp].map(lambda sentence: [word for word in sentence if word not in all_sw_ls])

    return X, all_sw_ls


# filter the postags, lemmatize
def nlp_lemmatization(X=None, col_nlp='col_nlp', enable_tag_filter=True, custom_mapper=True,
                      allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')):
    """
    Input:
    texts =['aa', 'bb']
    global_nlp
    """

    # sentence_lemma = None
    sentence_list_lemma = []

    sentence_list = X[col_nlp].tolist()

    for sentence in sentence_list:
        doc = global_nlp(" ".join(sentence))

        if enable_tag_filter:
            sentence_lemma = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        else:
            sentence_lemma = [token.lemma_ for token in doc]

        # custom mapper
        if custom_mapper:
            sentence_lemma = [iCUSTOM_WORD_LEMMA.get(item, item) for item in sentence_lemma]

        sentence_list_lemma.append(sentence_lemma)

    X[col_nlp] = pd.Series(sentence_list_lemma)

    return X


def generate_token_data_advance(data=None, col_main=None, idict_SW_mapper=None, col_nlp='col_nlp', punc_re=None,
                                deacc=False,
                                use_nltk=True, use_spacy=True, enable_tag_filter=True,
                                allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV'),
                                bigram_common_terms=(), bigrm_min_count=1, bigrm_threshold=0.1
                                ):
    X = data.copy()
    X[col_nlp] = X[col_main].copy()

    # get lower/clean/token data
    X = nlp_to_lowercase(X=X, col_nlp=col_nlp)
    X = nlp_remove_punctuation(X=X, col_nlp=col_nlp, punc_re=punc_re)
    X = sent_to_words(X=X, deacc=deacc, col_nlp=col_nlp)

    # --------------------------------------------------------------------------------------------------------------------------------
    # exact order is vvvv Imp  : stopword --> lemma --> bigram

    # custom_sw = MANNUAL_STOP_WORDS + idict_SW_mapper[col_main]
    # X, all_sw_ls = nlp_filter_stopwords(X=X, col_nlp=col_nlp, use_nltk=use_nltk, use_spacy=use_spacy, custom_sw=custom_sw)

    # lemma : no need to filter sw before lemma
    X = nlp_lemmatization(X=X, col_nlp=col_nlp, enable_tag_filter=enable_tag_filter, allowed_postags=allowed_postags)

    # now I have very clean/lemma words, good to filter sw BUT only the std ones
    custom_sw = []
    X, all_sw_ls = nlp_filter_stopwords(X=X, col_nlp=col_nlp, use_nltk=use_nltk, use_spacy=use_spacy,
                                        custom_sw=custom_sw)
    # --------------------------------------------------------------------------------------------------------------------------------

    # I kept custom_sw till this point to get bigram out of them
    X = nlp_make_bigram(X=X, col_nlp=col_nlp, min_count=bigrm_min_count, threshold=bigrm_threshold,
                        common_terms=bigram_common_terms)

    # now since I have made max out of data, it is good to filter out custom SW
    # to incr the speed as they are already filtered out, use_nltk=False, use_spacy=False
    custom_sw = MANNUAL_STOP_WORDS + idict_SW_mapper[col_main]
    X, all_sw_ls = nlp_filter_stopwords(X=X, col_nlp=col_nlp, use_nltk=use_nltk, use_spacy=use_spacy,
                                        custom_sw=custom_sw)

    return X, all_sw_ls


def generate_pp_data(data=None, col_main=None, col_pk=None, dict_ques_mapper=None):
    X = data.copy()
    X = X.rename(columns=dict_ques_mapper)
    X = X[[col_pk, col_main]]

    X = X[X[col_main] != NA_DATA]
    X = X.dropna()
    X = X.reset_index(drop=True)

    return X


def print_token_info(data=None, col_nlp='col_nlp', num_bigram_print=-1):
    X = data.copy()
    arr = X[col_nlp].tolist()
    arr = list(set([i for j in arr for i in j]))
    arr_bi = [i for i in arr if '_' in i]

    print("num docs        :", len(X))
    print("num unique kw   :", len(arr))
    print("num bi-grams kw :", len(arr_bi))

    print("\nBigarms: ")
    if num_bigram_print == -1:
        print(arr_bi)
    else:
        print(arr_bi[:num_bigram_print])
