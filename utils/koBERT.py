from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import pandas as pd
import numpy as np

model, vocab = get_pytorch_kobert_model()

class BERTDataset(Dataset):
      def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                  pad, pair):
          transform = nlp.data.BERTSentenceTransform(
              bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair) 

          self.sentences = [transform([i[sent_idx]]) for i in dataset]
          self.labels = [np.int32(i[label_idx]) for i in dataset]

      def __getitem__(self, i):
          return (self.sentences[i] + (self.labels[i], ))

      def __len__(self):
          return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                bert,
                hidden_size = 768,
                num_classes = 6, # softmax 사용 <- binary일 경우는 2
                dr_rate=None,
                params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

####################################################

def kobert_sent(text, vocab, model, model_path):

  device = torch.device("cuda:0")
  tokenizer = get_tokenizer()
  tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

  # dict
  small_sent_idx_to_dict ={0: '기쁜', 1: '스트레스 받는', 2: '당황', 3: '편안한', 4: '불안', 5: '우울한', 6: '짜증내는', 7: '분노', 8: '상처', 9: '열등감', 10: '죄책감의', 11: '당혹스러운', 12: '두려운', 13: '실망한', 14: '염세적인', 15: '충격 받은', 16: '슬픔', 17: '억울한', 18: '외로운', 19: '괴로워하는', 20: '자신하는', 21: '질투하는', 22: '감사하는', 23: '후회되는', 24: '툴툴대는'}
  sm_sent_big_sent_dict = {'기쁜' : '기쁨', '편안한' : '기쁨', '자신하는' : '기쁨', '감사하는' : '기쁨', '외로운' : '당황', '스트레스 받는' : '불안', '열등감' : '당황', '당황' : '당황', '죄책감의' : '당황', '짜증내는' : '분노', '분노' : '분노', '툴툴대는' : '분노', '당혹스러운' : '불안', '불안' : '불안', '두려운' : '불안', '괴로워하는' : '상처', '상처' : '상처', '충격 받은' : '상처', '억울한' : '상처', '질투하는' : '상처', '염세적인' : '슬픔', '슬픔' : '슬픔', '실망한' : '슬픔', '우울한' : '슬픔', '후회되는' : '슬픔'}
  sm_en_dic = {'슬픔' : 'Sadness','상처' : 'Hurt',  '툴툴대는' : 'Grumbling',  '자신하는' : 'Confident',  '당혹스러운' : 'Baffling',  '스트레스 받는' : 'Stressed',  '질투하는' : 'Envy',  '편안한' : 'Comfortable',  '실망한' : 'Disappointed',  '죄책감의' : 'Guilty',  '염세적인' : 'Pessimistic',  '기쁜' : 'Happy',  '열등감' : 'Inferiority',  '불안' : 'Anxiety',  '억울한' : 'Unfair',  '감사하는' : 'Grateful',  '괴로워하는' : 'Distressed',  '후회되는' : 'Regretful',  '당황' : 'Emberrassed',  '충격 받은' : 'Shocked',  '우울한' : 'Gloomy',  '분노' : 'Rage',  '짜증내는' : 'Annoying',  '두려운' : 'Scared',  '외로운' : 'Lonely',   '기쁨' : 'Happy',  '슬픔' : 'Sadness'}

  # model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
  sent_idx_to_dict_list = []

  # 입력값 확인
  try :
    test_sentence = str(text)
  except :
    print('문장을 확인하세요')
    pass

  # 감정 분류 확인 후 max_len, dict 지정
  max_len = 25
  sent_dict = small_sent_idx_to_dict

  # 데이터 형식 처리
  unseen_test = pd.DataFrame([[test_sentence, 5]], columns = [['text', 'label']])
  unseen_values = unseen_test.values
  test_set = BERTDataset(unseen_values, 0, 1, tok, max_len, True, False)
  test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0)

  #load_model
  model = model
  model = torch.load(model_path)


  for batch_id, (token_ids, valid_length, segment_ids, _) in enumerate(test_input):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length
    out = model(token_ids, valid_length, segment_ids)
    # print('예측감정 :', sent_dict[int(out[0].argmax().cpu().numpy())] , '\n', '실제감정 :', sent_dict[test_label])
    sent_idx_to_dict_list.append(sent_dict[int(out[0].argmax().cpu().numpy())])

  return sm_sent_big_sent_dict[sent_idx_to_dict_list[0]] , sm_en_dic[sm_sent_big_sent_dict[sent_idx_to_dict_list[0]]] , sent_idx_to_dict_list[0] , sm_en_dic[sent_idx_to_dict_list[0]]