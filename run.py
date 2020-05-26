import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sbn
import os
from functools import cmp_to_key
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from deepctr.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DeepFM
import pickle
import time

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_dict_from_file(file, type='default', skiprow=0):
    data = list()
    size = 0
    with open(file, encoding='utf-8') as f:
        for line in f:
            size += 1
            try:
                if size <= skiprow:
                    continue
                line = line.replace('"', '').strip()
                if len(line) <= 0:
                    continue
                if type == 'interet':
                    parts = line.split(",")
                    name = parts[0]
                    size = int(parts[1])
                    if size > 10 and len(name) > 0:
                        data.append(name)
                elif type == 'loc':
                    parts = line.split(",")
                    province = parts[0]
                    city = parts[1]
                    area = parts[2]
                    data.append(province)
                    data.append(province + "_" + city)
                    data.append(province + "_" + city + "_" + area)
                elif type == 'publisher':
                    parts = line.split(",")
                    name = parts[0]
                    size = int(parts[1])
                    if size > 10:
                        data.append(name)
                else:
                    data.append(line)
            except:
                print(line)
    return data


def load_dict(dir, type='default', skiprow=0):
    data = list()
    files = os.listdir(dir)
    for file in files:
        if not file.endswith(".csv"):
            continue
        data += load_dict_from_file(dir + "/" + file, type, skiprow)
    return data


lbe_file = 'lbe.pickle'
data_map_file = 'data_map.pickle'


def save_pickle_data(file, data):
    f = open(file, 'wb')
    pickle.dump(data, f)
    f.close()


def load_pickle_data(file):
    try:
        f1 = open(file, 'rb')
        return pickle.load(f1)
    except:
        pass
    return None


if os.path.exists(lbe_file) and os.path.exists(data_map_file):
    print('load data from cache')
    lbe_pickle = load_pickle_data(lbe_file)
    data_map_pickle = load_pickle_data(data_map_file)

    if lbe_pickle:
        lbes = lbe_pickle
    if data_map_pickle:
        rschannlemap = data_map_pickle['rschannlemap']
        itsmap = data_map_pickle['itsmap']
        locmap = data_map_pickle['locmap']
        vocabs = data_map_file['vocabs']
else:
    its = load_dict('/home/recsys/dataset/dict/interets', 'interet')
    locs = load_dict('/home/recsys/dataset/dict/loc', 'loc')
    publishers = load_dict('/home/recsys/dataset/dict/publisher', 'publisher', 1)
    cates = load_dict_from_file('/home/recsys/dataset/dict/cate.csv', 'cate', 1)
    channels = load_dict_from_file('/home/recsys/dataset/dict/channel.csv', 'channel', 1)
    publishers.append('other')
    channels.append('')

    u_levels = [str(i) for i in range(0, 10)]
    media_levels = [str(i) for i in range(0, 10)]

    vocabs = dict()
    vocabs['u_level'] = u_levels
    vocabs['t_channel'] = channels
    vocabs['cp_l1_category'] = cates
    vocabs['cp_publisher'] = publishers
    vocabs['cp_media_level'] = media_levels


    def gen_label_encode(vocab):
        lbe = LabelEncoder()
        lbe.fit(vocab)
        return lbe


    lbes = dict()
    for key in vocabs.keys():
        lbes[key] = gen_label_encode(vocabs[key])

    for key in vocabs.keys():
        print(len(vocabs[key]))


    def gen_dict_map(vocad):
        its_index = dict()
        size = 0
        for i in vocad:
            if i not in its_index.keys():
                its_index[i] = size
                size += 1
        return its_index


    rschannles = [str(i) for i in range(1, 33)]
    rschannlemap = gen_dict_map(rschannles)

    itsmap = gen_dict_map(its)
    locmap = gen_dict_map(locs)

    data_map = {}
    data_map['rschannlemap'] = rschannlemap
    data_map['itsmap'] = itsmap
    data_map['locmap'] = locmap
    data_map['vocabs'] = vocabs
    save_pickle_data(lbe_file, lbes)
    save_pickle_data(data_map_file, data_map)

dir = '/home/recsys/dataset/train_csv'


def list_sort_files(dir):
    def compare(x, y):
        stat_x = os.stat(dir + "/" + x)
        stat_y = os.stat(dir + "/" + y)
        if stat_x.st_ctime < stat_y.st_ctime:
            return -1
        elif stat_x.st_ctime > stat_y.st_ctime:
            return 1
        else:
            return 0

    items = os.listdir(dir)
    items.sort(key=cmp_to_key(compare))
    return items


def load_corpus(path):
    files = os.listdir(path)
    final_file = None
    for file in files:
        if file.endswith('.csv'):
            final_file = path + "/" + file
            break
    if not final_file:
        return None
    print('load data from ', final_file)
    return pd.read_csv(final_file)


def parser_publisher(item):
    if item not in publishers:
        item = 'other'
    return item


def trans_data(item):
    item['u_umi'] = item['u_umi'].fillna('')
    item['u_umi_weight'] = item['u_umi_weight'].astype('str').fillna('')
    item['u_uli'] = item['u_uli'].fillna('')
    item['u_uli_weight'] = item['u_uli_weight'].astype('str').fillna('')
    item['u_usi'] = item['u_usi'].fillna('')
    item['u_usi_weight'] = item['u_usi_weight'].astype('str').fillna('')
    item['u_level'] = item['u_level'].fillna(0).astype('int').astype('str')
    item['cp_media_level'] = item['cp_media_level'].fillna(0).astype('int').astype('str')
    item['cp_location'] = item['cp_location'].fillna("")
    item['rs_channel'] = item['rs_channel'].fillna("")
    item['cp_interests'] = item['cp_interests'].fillna("")
    item['rs_p1_score'] = item['rs_p1_score'].fillna(0)
    item['rs_gactr'] = item['rs_gactr'].fillna(0)
    item['rs_taginfo'] = item['rs_taginfo'].fillna("")
    item['rs_taginfo_weight'] = item['rs_taginfo_weight'].fillna("")
    item['rs_dactr'] = item['rs_dactr'].fillna("")
    item['cp_publisher'] = item['cp_publisher'].apply(parser_publisher)
    item['t_channel'] = item['t_channel'].fillna("")


def gen_pad_seq(values, weights, key2index, max_len=None):
    def split(x):
        vkeys = list()
        try:
            key_ans = x.split(',')
            for key in key_ans:
                if key in key2index:
                    vkeys.append(key)
        except:
            pass
        return list(map(lambda x: key2index[x], vkeys))
    def split_weight(x):
        res = list()
        size = 0
        parts = x.split(',')
        for part in parts:
            try:
                if part != 'nan':
                    res.append(float(part))
                else:
                    res.append(0)
                size += 1
            except:
                res.append(0)
                size += 1
        while size < max_len:
            res.append(0)
            size += 1
        return res
    index_list = list(map(split, values))
    weight_list = None
    if weights is not None:
        weight_list = list(map(split_weight, weights))
    index_list = pad_sequences(index_list, maxlen=max_len, padding='post', )
    return index_list, weight_list


sparse_features = ['u_level', 't_channel', 'cp_l1_category', 'cp_publisher', 'cp_media_level']
dense_features = ['rs_gactr']
target = ['action']

files = list_sort_files(dir)

uli_len = 500
umi_len = 500
usi_len = 100
rs_tag_len = 10
cp_i_len = 10
cp_loc_len = 2
t_loc_len = 2
rs_channel_len = 32

var_info = list()
var_info.append({'label':'u_uli', 'len':uli_len, 'map':itsmap, 'weight':'u_uli_weight'})
var_info.append({'label':'u_umi', 'len':umi_len, 'map':itsmap, 'weight':'u_umi_weight'})
var_info.append({'label':'u_usi', 'len':usi_len, 'map':itsmap, 'weight':'u_usi_weight'})
var_info.append({'label':'rs_taginfo', 'len':rs_tag_len, 'map':itsmap, 'weight':'rs_taginfo_weight'})
var_info.append({'label':'cp_interests', 'len':cp_i_len, 'map':itsmap, 'weight':None})
var_info.append({'label':'cp_location', 'len':cp_loc_len, 'map':locmap, 'weight':None})
var_info.append({'label':'t_location', 'len':t_loc_len, 'map':locmap, 'weight':None})
var_info.append({'label':'rs_channel', 'len':rs_channel_len, 'map':rschannlemap, 'weight':None})

# define model
emb_size = 32
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=len(vocabs[feat]), embedding_dim=emb_size)
                              for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]
varlen_feature_columns = [VarLenSparseFeat(SparseFeat(item['label'], vocabulary_size=len(item['map']) + 1, embedding_dim=emb_size), maxlen=item['len'], combiner='mean', weight_name=item['weight']) for item in var_info]
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'binary_accuracy', tf.keras.metrics.AUC()])


def train_by_batch(file):
    train_corpus = load_corpus('/home/recsys/dataset/train_csv/' + file + '/train')
    test_corpus = load_corpus('/home/recsys/dataset/train_csv/' + file + '/test')
    choose_data = pd.concat([train_corpus, test_corpus])
    cur = int(time.time())
    trans_data(choose_data)
    print("trans_data cost", (int(time.time()) - cur))
    for feat in sparse_features:
        lbe = lbes[feat]
        choose_data[feat] = lbe.transform(choose_data[feat])
    cur = int(time.time())
    rs_channel_list, _ = gen_pad_seq(choose_data['rs_channel'], None, rschannlemap, 32)
    uli_list, uli_list_weight = gen_pad_seq(choose_data['u_uli'], choose_data['u_uli_weight'], itsmap, 500)
    umi_list, umi_list_weight = gen_pad_seq(choose_data['u_umi'], choose_data['u_umi_weight'], itsmap, 500)
    usi_list, usi_list_weight = gen_pad_seq(choose_data['u_usi'], choose_data['u_usi_weight'], itsmap, 100)
    rs_tag_list, rs_tag_weight = gen_pad_seq(choose_data['rs_taginfo'], choose_data['rs_taginfo_weight'], itsmap, 10)
    cp_i_list, _ = gen_pad_seq(choose_data['cp_interests'], None, itsmap, 10)
    cp_loc_list, _ = gen_pad_seq(choose_data['cp_location'], None, locmap, 2)
    t_loc_list, _ = gen_pad_seq(choose_data['t_location'], None, locmap, 2)
    print("trans_data cost", (int(time.time()) - cur))
    var_data = list()
    var_data.append({'label': 'u_uli', 'list': uli_list, 'weight': uli_list_weight})
    var_data.append({'label': 'u_umi', 'list': umi_list, 'weight': umi_list_weight})
    var_data.append({'label': 'u_usi', 'list': usi_list, 'weight': usi_list_weight})
    var_data.append({'label': 'rs_taginfo', 'list': rs_tag_list, 'weight': rs_tag_weight})
    var_data.append({'label': 'cp_interests', 'list': cp_i_list})
    var_data.append({'label': 'cp_location', 'list': cp_loc_list})
    var_data.append({'label': 't_location', 'list': t_loc_list})
    var_data.append({'label': 'rs_channel', 'list': rs_channel_list})
    print(uli_len, umi_len, usi_len, rs_tag_len, cp_i_len, cp_loc_len, t_loc_len)

    model_input = {name: choose_data[name] for name in feature_names}
    for item in var_data:
        model_input[item['label']] = item['list']
        if item['label'] == 'rs_taginfo':
            model_input[item['label'] + '_weight'] = item['weight']
        elif item['label'] == 'u_uli' or item['label'] == 'u_umi' or item['label'] == 'u_usi':
            model_input[item['label'] + '_weight'] = item['weight']

    history = model.fit(model_input, choose_data[target].values,
                        batch_size=64, epochs=5, verbose=2, validation_split=0.1)
    model.save_weights('./checkpoints/' + file)
    
def find_last_file(file):
    dt = ''
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            dt = line.replace('model_checkpoint_path: "', '').replace('"', '')
            break
    return dt


last_stmp = find_last_file('checkpoints/checkpoint')
isFind = False
for file in files:
    if len(last_stmp) <= 0:
        isFind = True
    if not isFind:
        if file != last_stmp:
            continue
        isFind = True
    else:
        print('train model use data from ', file)
        train_by_batch(file)

