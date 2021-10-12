#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import nltk
import spacy
import string
import gensim
import pypandoc
import pyLDAvis
import warnings
import shutil

import numpy  as np
import pandas as pd

import pyLDAvis.gensim_models as gensimvis
import gensim.corpora         as corpora
import matplotlib.pyplot      as plt

from gensim.corpora.dictionary import Dictionary
from gensim.test.utils         import datapath, common_texts

from wordcloud        import WordCloud
from nltk.corpus      import stopwords
from nltk.probability import FreqDist
from nltk.tokenize    import sent_tokenize, word_tokenize
from nltk.stem        import PorterStemmer, WordNetLemmatizer

from sklearn.neighbors               import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
pyLDAvis.enable_notebook()
warnings.filterwarnings("ignore")


# In[ ]:


sw_list  = stopwords.words('english')
sw_list += list(string.punctuation)
sw_list += ['©', '§', 'v.', '(see', 'The', 'Court,', 'New', 'York', 'In', 'Dept.',
            'A.D.3d', 'N.Y.S.2d', 'N.Y.S.3d', 'N.E.2d', 'N.Y.2d']
sw_set   = set(sw_list)


# In[ ]:


def make_db(file):
    
    f  = open(file, 'r', encoding='cp1252')
    f1 = f.readlines()
    
    start  = 0
    stop   = 0
    d_list = []    
    for line in f1[0:]:
        stop += 1
        if 'DECISION' in line:
            start = stop
        if 'End of Document\n' in line:
            d_list = f1[start:stop-1]
            
    f3 = []        
    for x in d_list:
        if x != '\n':
            f3.append(x[:-1])
            
    decision = ''
    for x in f3:
        if x != ' ':
            decision += str(x)
    
    return decision


# In[ ]:


def clean_train_data(source, outputfile):
    
    source = source
    os.chdir(source)

    name_list     = []
    decision_list = []

    for file in glob.glob("*.txt"):
        name_list.append(file[:-4])
        decision_list.append(make_db(file))

    df1 = pd.DataFrame(name_list, columns=['Name'])
    df2 = pd.DataFrame(decision_list, columns=['Decision'])
    df  = pd.concat([df1, df2], axis=1)
    
    pd.DataFrame.to_csv(df, outputfile)
    


# In[ ]:


def docx_to_text(source, outputfile):
    
    docxFilename = source
    output = pypandoc.convert_file(docxFilename, 'plain', outputfile=outputfile)
    assert output == ""
    


# In[ ]:


def make_input(file):
    
    a  = open(file, 'r')
    a1 = a.readlines()
    
    start  = 0
    stop   = 0
    i_list = []    
    for line in a1[0:]:
        stop += 1
        if 'Plaintiff,' in line:
            start = stop
        if 'PLEASE' in line:
            i_list = a1[start:stop-1]
            
    a3 = []        
    for x in i_list:
        if x != '\n':
            a3.append(x[:-1])
            
    file_input = ''
    for x in a3:
        if x != ' ':
            file_input += str(x)
            
    return file_input


# In[ ]:


def clean_target(data, outputfile):
    
    input_doc = make_input(data)
    columns   = ['Name', 'Text']
    Name      = ['Drew Benasillo v. Upper East Side Pain Medicine']
    Data      = {columns[1]:input_doc}
    Target    = pd.DataFrame(Data, Name, columns=columns)

    pd.DataFrame.to_csv(Target, outputfile)
    


# In[ ]:


def process_words(decisions, stop_words=sw_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    decisions     = [[word for word in doc.split() if word not in stop_words] for doc in decisions]
    decisions_out = []
    nlp           = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    
    for doc in decisions:
        doc = nlp(" ".join(doc)) 
        decisions_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
    decisions_out = [[word for word in doc if word not in stop_words] for doc in decisions_out]
    
    return decisions_out


# In[ ]:


def ldam(data, topics, save=False):
        
    data_ready = process_words(data)
    id2word    = corpora.Dictionary(data_ready)
    corpus     = [id2word.doc2bow(text) for text in data_ready]
    lda_model  = gensim.models.ldamodel.LdaModel(corpus       = corpus,
                                                 id2word      = id2word,
                                                 num_topics   = topics, 
                                                 random_state = 100,
                                                 update_every = 1,
                                                 chunksize    = 10,
                                                 passes       = 10,
                                                 alpha        = 'symmetric',
                                                 iterations   = 100,
                                                 per_word_topics = True)
        
    if save == True:
        temp_file = datapath('/Users/dimitrybelozersky/Documents/Phase5/lda model results/lda_model')
        lda_model.save(temp_file)
    
    vis = gensimvis.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    
    return vis, lda_model


# In[ ]:


def ldam_topics(target, new_model=False, model=False):
        
    temp_file = datapath('/Users/dimitrybelozersky/Documents/Phase5/lda model results/lda_model')
    lda_model = gensim.models.ldamodel.LdaModel.load(temp_file)
    
    if new_model == True:
        lda_model = model
    
    Target   = process_words(target)
    T_dict   = corpora.Dictionary(Target)
    T_corpus = [T_dict.doc2bow(text) for text in Target]

    target_doc = T_corpus[0]
    vector     = lda_model[target_doc]    
    
    topic_1_name  = '1: Negligence/Premises'
    topic_1_match = vector[0][0][1]

    topic_2_name  = '2: Medical Malpractice'
    topic_2_match = vector[0][1][1]

    X = [topic_1_name, topic_2_name]
    Y = [topic_1_match, topic_2_match]
    Y = np.array(Y)* 100

    plt.bar(X,Y)
    plt.title('Target Document Topic')
    plt.xlabel('Topics')
    plt.ylabel('Topic %')
    plt.show()    
    


# In[ ]:


def spacy_train_data(data, spacy_type, outputfile):
    
    if spacy_type == 'md':
        spacy_nlp       = spacy.load('en_core_web_md')
    if spacy_type == 'lg':
        spacy_nlp       = spacy.load('en_core_web_lg')
    if spacy_type == 'trf':
        spacy_nlp       = spacy.load('en_core_web_trf')
    
    df['Spacy']     = data.apply(lambda x: spacy_nlp(x))
    Train_spacy_out = pd.DataFrame(np.vstack([x.vector for x in df.Spacy]))
    pd.DataFrame.to_csv(Train_spacy_out, outputfile)


# In[ ]:


def docummend(target, vectorizer, results, out=False, Topics=False, LDA=False):
    
    if vectorizer == 'spacy':
        Train_data      = pd.read_csv('data_csv/Train_spacy_out.csv', index_col=False)
        Train_data      = Train_data.drop(['Unnamed: 0'], axis=1)
        spacy_nlp       = spacy.load('en_core_web_md')
        target['Spacy'] = target.Text.apply(lambda x: spacy_nlp(x))
        Target_data     = pd.DataFrame(np.vstack([x.vector for x in target.Spacy]))
        
    if vectorizer == 'spacy lg':
        Train_data      = pd.read_csv('data_csv/Train_spacy_out_lg.csv', index_col=False)
        Train_data      = Train_data.drop(['Unnamed: 0'], axis=1)
        spacy_nlp       = spacy.load('en_core_web_lg')
        target['Spacy'] = target.Text.apply(lambda x: spacy_nlp(x))
        Target_data     = pd.DataFrame(np.vstack([x.vector for x in target.Spacy]))       
        
    if vectorizer == 'spacy acc':
        Train_data      = pd.read_csv('data_csv/Train_spacy_out_accuracy.csv', index_col=False)
        Train_data      = Train_data.drop(['Unnamed: 0'], axis=1)
        spacy_nlp       = spacy.load('en_core_web_trf')
        target['Spacy'] = target.Text.apply(lambda x: spacy_nlp(x))
        Target_data     = pd.DataFrame(np.vstack([x.vector for x in target.Spacy]))
    
    if vectorizer == 'tfidf':
        Train_data  = pd.read_csv('data_csv/Decision_Database.csv', index_col=False)
        Train_data  = Train_data.drop(['Unnamed: 0','length', 'Name'], axis=1)
        tfidf_nlp   = TfidfVectorizer(stop_words=stopwords.words('english'))
        Train_data  = tfidf_nlp.fit_transform(Train_data['Decision'])
        Target_data = tfidf_nlp.transform(target['Text'])
        
    if vectorizer == 'cv':
        Train_data  = pd.read_csv('data_csv/Decision_Database.csv', index_col=False)
        Train_data  = Train_data.drop(['Unnamed: 0','length', 'Name'], axis=1)     
        cv_nlp      = CountVectorizer(stop_words=sw_set)
        Train_data  = cv_nlp.fit_transform(Train_data['Decision'])
        Target_data = cv_nlp.transform(target['Text'])
    
    neigh  = NearestNeighbors()
    neigh.fit(Train_data)
    result = neigh.kneighbors(Target_data, n_neighbors=results)
 
    df = pd.read_csv('data_csv/Decision_Database.csv', index_col=False)
    df = df.drop(['Unnamed: 0', 'length'], axis=1)
    
    distance       = [x for x in result[0].tolist()]
    decision_index = [x for x in result[1].tolist()]
    decision_name  = [df['Name'].iloc[x] for x in decision_index[0]]
    decision_text  = [df['Decision'].iloc[x] for x in decision_index[0]]       
    result_dict    = dict(zip(distance[0], decision_name))
    percent_array  = np.array(distance) * 100
    number         = 0
    
    print('\nTop ' + str(results) + ' Recommended Decisions - ' + vectorizer + ':\n')
    for x in result_dict.values():
        number += 1
        print(str(number) + '.  '+ x)
        
    if vectorizer == 'spacy':
        percent_array  = np.array(distance) * 100
        plt.xlabel('% Difference From Target Document')
    if vectorizer == 'spacy lg':
        percent_array  = np.array(distance) * 100
        plt.xlabel('% Difference From Target Document')
    if vectorizer == 'tfidf':
        percent_array  = np.array(distance)
        plt.xlabel('Distance Difference From Target Document')
    if vectorizer == 'cv':
        percent_array  = np.array(distance)
        plt.xlabel('Distance Difference From Target Document')
        
    plt.barh(decision_name, percent_array[0])
    plt.title('Distance Of Results From Target Document')
    plt.show()
    
    if out == True:
        
        doc_source  = '/Users/dimitrybelozersky/Documents/Phase5/all_decision_data_txt/'
        dest        = '/Users/dimitrybelozersky/Documents/Phase5/nn_model_results/'
        
        for name in decision_name:
            pattern = name + '.txt'
            
            files = glob.glob(doc_source + pattern)
        
            for file in files:
                file_name = os.path.basename(file)
                shutil.copy(file, dest + file_name)
    
    if Topics == True:
        
        ldam_topics(target.Text)
        
    if LDA == True:
        
        decision_df1  = pd.DataFrame(decision_name, columns=['Name'])
        decision_df2  = pd.DataFrame(decision_text, columns=['Decision'])
        decision_df   = pd.concat([decision_df1, decision_df2], axis=1)         
        output, model = ldam(decision_df['Decision'], results)
        return output
    
    else:    
        pass 


# In[ ]:




