import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def jaccard_similarity(data, question):
    
    union = set(data).union(set(question))
    intersection = set(data).intersection(set(question))

    jaccard_sim = len(intersection) / len(union)  

    return jaccard_sim

def jaccard_high(data, question, num, col):
    
    # data: 데이터프레임, question: 입력한 텍스트(질문), num: 자카드 유사도 상위 갯수
    data['jaccard_similarity'] = data[col].apply(lambda x: jaccard_similarity(x, question))

    return data[['video', 'big_sent', 'sm_sent', 'sent', 'action', 'jaccard_similarity']].sort_values(['jaccard_similarity'], ascending=False)[:num]

def tokenized_output(tokens):
    return tokens

def cos_similarity(data, question, col):

    tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                       tokenizer=tokenized_output,
                                       preprocessor=tokenized_output,
                                       token_pattern=None)
    
    tfidf_data = tfidf_vectorizer.fit_transform(data[col])
    tfidf_question = tfidf_vectorizer.transform([question])

    data['cosine_similarity'] = cosine_similarity(tfidf_data, tfidf_question).reshape(-1, )

    return data[['video', 'big_sent', 'sm_sent', 'sent', 'action','jaccard_similarity', 'cosine_similarity']].sort_values(['cosine_similarity'], ascending=False)

def video_rec(sent_token, act_token, data):
  token_list = []
  for i in range(len(data)):
    bg_list = []
    for bg in data.loc[i, 'big_sent'][1:-1].replace("'", '').split(',') :
      bg_list.append(bg.strip())
    for sm in data.loc[i, 'sm_sent'][1:-1].replace("'", '').split(',') :
      bg_list.append(sm.strip())
    token_list.append(list(np.unique(np.array(bg_list))))
  data['sent'] = token_list

  ques_token = sent_token
  result = jaccard_high(data, ques_token, 20, 'sent')
  result = cos_similarity(result, ques_token, 'sent')
  sent_cos_list = set(result.video[:20].values)

  ques_token = act_token
  result = jaccard_high(data, ques_token, 20, 'action')
  result = cos_similarity(result, ques_token, 'action')
  act_cos_list = set(result.video[:20].values)

  rec_list = list(sent_cos_list.intersection(act_cos_list))
  if len(rec_list) == 1:
    print('추천 비디오가 없습니다')
  else:
    return rec_list

def video_infos(rec_list):
    dscript_lst = []
    for name in rec_list:
        json_object = json.load(open(f'./static/multi_dataset/{name}/{name}_interpolation.json', encoding='utf-8'))
        dscript = json_object['common_info']['clip_descs'][0]
        dscript_lst.append(dscript)
    return dscript_lst