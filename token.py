import torch
import numpy as np
import pandas as pd

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
'''
from transformers import AutoTokenizer
from transformers import TFBertForMaskedLM
'''
# train dataset
train = pd.read_csv("ratings.txt", sep = "\t")

#리뷰 본문만 추출
sentence = train['document']

# [CLS]는 문장의 시작에 사용
# [SEP]는 문장의 끝이나 두 문장 분리에 사용
sentence = ["[CLS] " + str(i) + " [SEP]" for i in sentence ]

#라벨 추출
labels = train['label'].values

# Bert-base의 토크나이저로 문장을 토큰으로 분리
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenized = [tokenizer.tokenize(i) for i in sentence]

# 토큰을 숫자 인덱스로 변환
input = [tokenizer.convert_tokens_to_ids(i) for i in tokenized]

# 문장을 max_len 길이에 맞게 자르고 모자란 부분은 0으로 채움
input = pad_sequences(input, maxlen = 50, dtype = "long", truncating = "post", padding = "post")

# Attention mask 생성
# 패딩에 해당하는 부분은 0으로 패딩이 아닌 부분은 1로 간주하는 마스크 생성
# 어텐션 마스크 초기화
attention_masks = []

for se in input:
    seq_mask = [float(i>0) for i in se]
    attention_masks.append(seq_mask)

# 데이터를 텐서로 변환
# train 비율과 validation의 비율이 9대 1
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input,
                                                                                    labels, 
                                                                                    random_state=0, 
                                                                                    test_size=0.1)

# 어텐션 마스크를 훈련셋과 검증셋으로 분리
train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input,
                                                       random_state=0, 
                                                       test_size=0.1)

# 데이터를 파이토치의 텐서로 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = 50)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=50)

#test dataset
df = pd.read_csv('output.csv',sep = ",")

#good(1) bad(0) 라벨붙이기
df2 = pd.DataFrame(df, columns=['label'])

def rating_label(rating):
    if rating > 3:
        return 1   #good
    else:
        return 0   #bad
df2['label'] = df['star'].apply(lambda x: rating_label(x))
test = pd.concat([df,df2],axis = 1)

#test data 리뷰 추출
test_sentence = test['content']
test_sentence = ["[CLS] " +str(i) + " [SEP]" for i in test_sentence]
test_label = test['label']

# Bert-base의 토크나이저로 문장을 토큰으로 분리
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenized = [tokenizer.tokenize(i) for i in test_sentence]

# 토큰을 숫자 인덱스로 변환
input = [tokenizer.convert_tokens_to_ids(i) for i in tokenized]

# 문장을 max_len 길이에 맞게 자르고 모자란 부분은 0으로 채움
input = pad_sequences(input, maxlen = 200, dtype = "long", truncating = "post", padding = "post")

# Attention mask 생성
# 패딩에 해당하는 부분은 0으로 패딩이 아닌 부분은 1로 간주하는 마스크 생성
# 어텐션 마스크 초기화
attention_masks = []

for se in input:
    seq_mask = [float(i>0) for i in se]
    attention_masks.append(seq_mask)

test_inputs = torch.tensor(input)
test_labels = torch.tensor(test_label)
test_masks = torch.tensor(attention_masks)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = 40)


# 디바이스 설정
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

# BERT 모델 생성
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()

# optimizer 설정 => optimizer는 AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)

epochs = 5

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

  #validation
  model.eval()
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
      
    with torch.no_grad():
      output = model(b_input_ids, token_type_ids=None, attention_mask = b_input_mask)
      logits = output[0]
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      pred_flat = np.argmax(logits, axis=1).flatten()
      labels_flat = label_ids.flatten()
      tmp_eval_accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
      eval_accuracy += tmp_eval_accuracy

      nb_eval_steps += 1
  print("Validation Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))


# test

model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
pro, con = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)
    
    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch
    
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    
    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    #정확도 계산
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    
    for i in range(len(pred_flat)):
      if (pred_flat[i] == labels_flat[i]):
        if (pred_flat[i] == 1):
          pro += 1
        else:
          con += 1
    
    tmp_eval_accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
print("긍정으로 판단: ", pro)
print("부정으로 판단: ", con)
