import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
import pandas as pd
from tqdm import *
from sklearn.preprocessing import MinMaxScaler

# 加载预训练模型和分词器
config_path = r'D:\My_Files\predict_module\uncased_L-24_H-1024_A-16\bert_config.json'
checkpoint_path = r'D:\My_Files\predict_module\uncased_L-24_H-1024_A-16\bert_model.ckpt'
dict_path = r'D:\My_Files\predict_module\uncased_L-24_H-1024_A-16\vocab.txt'
maxlen = 256

# 这里with_mlm 打开预测单字模式
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 定义好模板
# prefix = 'mask, '
# mask_idxs = [0]

# 定义类别
labels = [
    'rarely', 'sometimes', 'often', 'always'
]

data = pd.read_csv(r'E:\SUGAR-master\dataset\best\1\Raw\depression.txt', header=None, sep='\t')
sds = pd.read_csv(r'sds.txt', header=None, sep='\t')
num = 0
ans1 = []

post = 'Today I feel that I want to learn and fun and happy.I LOVE TODAY.' \
       'Tomorrow I hope it rains because I want to play in the class room with Rita.' \
       'Today one thing that surprised me today was there are a... NEW 3 STUDENTS. YAY!!!' \
       'Today I do it science i\'m do it little be it help.' \
       'Today at P.E we do don\'t let our con in our team fell. And we do a team is boy v.s girl.' \
       'Today I had P.E boy v.s girl on doge ball castle it was fun!' \
       'Today I will have a airsoft gun Fusheng.' \
       'Today I went to the wax museum and I saw the queen of Egypt Cleopatra  she was super cool.'


''''Bereaved families can plant snowdrops in remembrance of loved ones at annual walk.' \
       'These essential oils can help you when dealing with depression through a bereavement.' \
       '10 things I wish someone would have told me about #grief. #bereavement #counselling #support.' \
       '#JohnTravolta posts touching tribute about his #bereavement after son’s death.' \
       'Bucks Fizz star raises money for #bereavement suite.' \
       'Thought provoking quotes about #loss and #bereavement.' \
       'Coping with the loss of a loved one. '''

for j in range(20):
    mask_idxs = [-(len(sds.values[j][0].split(' ')) - sds.values[j][0].split(' ').index('mask'))]
    text = post + sds.values[j][0]
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids[mask_idxs[0]] = tokenizer._token_mask_id
    token_ids = sequence_padding([token_ids])
    segment_ids = sequence_padding([segment_ids])

    label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])
    y_pred = model.predict([token_ids, segment_ids])[:, mask_idxs]
    y_pred = y_pred[:, 0, label_ids[:, 0]]  # * y_pred[:, 1, label_ids[:, 1]]
    y_pred = y_pred.argmax(axis=1)
    print(labels[y_pred[0]])

'''
# for j in range(20):
j = 3
temp = [[0, 0], [0, 0], [0, 0], [0, 0]]
for i in tqdm(range(data.shape[0])):
    mask_idxs = [-(len(sds.values[j][0].split(' ')) - sds.values[j][0].split(' ').index('mask'))]
    text = data.values[i][2] + '. ' + sds.values[j][0]
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids[mask_idxs[0]] = tokenizer._token_mask_id
    token_ids = sequence_padding([token_ids])
    segment_ids = sequence_padding([segment_ids])

    label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])
    y_pred = model.predict([token_ids, segment_ids])[:, mask_idxs]
    y_pred = y_pred[:, 0, label_ids[:, 0]]  # * y_pred[:, 1, label_ids[:, 1]]
    y_pred = y_pred.argmax(axis=1)
    if data.values[i][1] == 'positive':
        temp[y_pred[0]][0] += 1
    else:
        temp[y_pred[0]][1] += 1
print(temp)'''
'''pd.DataFrame([temp]).to_csv('prompt2.txt', sep='\t', mode='a', header=False,
                                index=False,
                                encoding='UTF-8')'''

'''for i in tqdm(range(data.shape[0])):
    temp = [0, 0, 0, 0]
    for j in range(20):
        mask_idxs = [-(len(sds.values[j][0].split(' ')) - sds.values[j][0].split(' ').index('mask'))]
        text = data.values[i][2] + '. ' + sds.values[j][0]
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        token_ids[mask_idxs[0]] = tokenizer._token_mask_id
        token_ids = sequence_padding([token_ids])
        segment_ids = sequence_padding([segment_ids])

        label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])
        y_pred = model.predict([token_ids, segment_ids])[:, mask_idxs]
        y_pred = y_pred[:, 0, label_ids[:, 0]]  # * y_pred[:, 1, label_ids[:, 1]]
        y_pred = y_pred.argmax(axis=1)
        # print(labels[y_pred[0]])
        temp[y_pred[0]] += int(sds.values[j][y_pred[0] + 1])
    ans = sum(temp) / 80
    if (ans >= 0.5 and data.values[i][1] == 'positive') or (ans < 0.5 and data.values[i][1] == 'negative'):
        pred = True
        num += 1
    else:
        pred = False

    ans1.append(temp)


    def standardization(data1):
        mi = min(data1)
        ma = max(data1) - mi
        for k in range(len(data1)):
            data1[k] = round((data1[k] - mi) / ma, 3)
        return data1


    temp = standardization(temp)
    print(i + 1, data.values[i][1], temp, ans, pred)
    print('acc:', round(num / (i + 1), 3))
    pd.DataFrame([[i + 1, data.values[i][1], temp, ans, pred]]).to_csv('prompt1.txt', sep='\t', mode='a', header=False,
                                                                       index=False,
                                                                       encoding='UTF-8')
ans = np.array(ans1)
scaler = MinMaxScaler()
result = scaler.fit_transform(ans)
pd.DataFrame(result).to_csv('prompt.txt', sep='\t', mode='a', header=False,
                                                                       index=False,
                                                                       encoding='UTF-8')'''
