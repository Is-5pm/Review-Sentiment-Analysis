import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import random
import time
import csv

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import TFBertForMaskedLM

# 데이터 로드
# 열 이름 추가해서 데이터로드
#학습 데이터
train = pd.read_csv("ratings.txt", sep = "\t")
print(train.shape)

#테스트 데이터
# test_data = pd.read_csv("")
#print(test.shape)

#리뷰 본문만 출력
sentence = train['document']
#sentence[:10]

# [CLS]는 문장의 시작에 사용
# [SEP]는 문장의 끝이나 두 문장 분리에 사용

sentence = ["[CLS] " + str(i) + " [SEP]" for i in sentence ]
#sentence[:10]

#라벨 추출
# 우리는 데이터에 라벨을 추가해야함
labels = train['label'].values


# Bert-base의 토크나이저로 문장을 토큰으로 분리
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenized = [tokenizer.tokenize(i) for i in sentence]

# 토큰을 숫자 인덱스로 변환
input = [tokenizer.convert_tokens_to_ids(i) for i in tokenized]

# 문장을 max_len 길이에 맞게 자르고 모자란 부분은 0으로 채움
input = pad_sequences(input, maxlen = 50, dtype = "long", truncating = "post", padding = "post")
input[0]

# Attention mask 생성
# 패딩에 해당하는 부분은 0으로 패딩이 아닌 부분은 1로 간주하는 마스크 생성

# 어텐션 마스크 초기화
attention_masks = []

for se in input:
    seq_mask = [float(i>0) for i in se]
    attention_masks.append(seq_mask)

print(attention_masks[0])


# 데이터를 텐서로 변환
# train 비율과 validation의 비율이 9대 1
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input,
                                                                                    labels, 
                                                                                    random_state=2018, 
                                                                                    test_size=0.1)

# 어텐션 마스크를 훈련셋과 검증셋으로 분리
train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input,
                                                       random_state=2018, 
                                                       test_size=0.1)

# 데이터를 파이토치의 텐서로 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)				

train_inputs.shape, validation_inputs.shape, train_labels.shape, validation_labels.shape

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = 50)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=50)

# GPU 디바이스 이름 구함
#device_name = tf.test.gpu_device_name()
#print(device_name)
# GPU 디바이스 이름 검사
#if device_name == '/device:GPU:0':
#    print('Found GPU at: {}'.format(device_name))
#else:
#    raise SystemError('GPU device not found')

# 디바이스 설정
if torch.cuda.is_available():
  device = torch.device("cuda")
  print("cuda")
else:
  device = torch.device("cpu")
  print("cpu")

# BERT 모델 생성
# label을 2개로 할지 3개로 할지
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()

# optimizer 설정 => optimizer는 AdamW

optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)

epochs = 5

# 스케줄러 생성(학습률을 조금씩 감소)


train_iterator = iter(train_dataloader)
#dir(train_iterator)
a1 = next(train_iterator)

# 학습
for i in range(epochs):
 
  cost = 0
  model.train()
  print("train")
  print( i, " / ", epochs)

  #데이터로더에서 배치만큼 가져옴

  for step, batch in enumerate(train_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    result = model(b_input_ids, token_type_ids=None, attention_mask = b_input_mask, labels = b_labels)
    loss = result[0]
    cost +=loss.item()
    loss.backward()

    optimizer.step()
    model.zero_grad()
    train_cost = cost / len(train_dataloader)

    if step % 100 == 0:
        print("batch: ", step, "/ ", len(train_dataloader) )
        print("cost", train_cost)


  model.eval()
  eval_accuracy = 0
  for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
      
    with torch.no_grad():
      output = model(b_input_ids, token_type_ids=None, attention_mask = b_input_mask)

      loss2 = output[0]


# 테스트 