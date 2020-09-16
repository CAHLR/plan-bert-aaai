import sys
sys.path.append("..")
import os, pickle, importlib, tqdm, sys, keras, copy, Engine
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
import util.Generator as Generator
importlib.reload(Generator)
from util.Datahelper import UNIVERSITY1_load_data, SYSTEM1_load_data, list_partition, list_sampling, list_padding, mat_partition, mat_sampling, mat_padding, set_top_n, list2mat

tf.random.set_seed(0)
np.random.seed(0)

SYSTEM1_num = int(sys.argv[1])
#SYSTEM1_num = 13

SYSTEM1_feat_list = [
    {'dir':'../../../FULL_data/29614109','test_sem':9,'num_crs':1341,'user_feat':[8,365],'item_feat':[67, 4, 27, 5]},
    {'dir':'../../../FULL_data/29614110','test_sem':9,'num_crs':570 ,'user_feat':[8,365],'item_feat':[79, 5, 20, 1]},
    {'dir':'../../../FULL_data/29614111','test_sem':9,'num_crs':2173,'user_feat':[8,365],'item_feat':[66, 12, 49, 9]},
    {'dir':'../../../FULL_data/29614134','test_sem':9,'num_crs':879 ,'user_feat':[8,365],'item_feat':[68, 4, 19, 1]},
    {'dir':'../../../FULL_data/29614113','test_sem':9,'num_crs':1636,'user_feat':[8,365],'item_feat':[69, 3, 30, 7]},
    {'dir':'../../../FULL_data/29614114','test_sem':9,'num_crs':3437,'user_feat':[8,365],'item_feat':[77, 5, 44, 11]},
    {'dir':'../../../FULL_data/29614115','test_sem':9,'num_crs':419 ,'user_feat':[8,365],'item_feat':[59, 9, 13, 2]},
    {'dir':'../../../FULL_data/29614116','test_sem':9,'num_crs':3538,'user_feat':[8,365],'item_feat':[105, 14, 39, 8]},
    {'dir':'../../../FULL_data/29614117','test_sem':9,'num_crs':1281,'user_feat':[8,365],'item_feat':[67, 6, 27, 3]},
    {'dir':'../../../FULL_data/29614118','test_sem':6,'num_crs':769 ,'user_feat':[8,365],'item_feat':[69, 7, 17, 2]},
    {'dir':'../../../FULL_data/29614119','test_sem':6,'num_crs':769 ,'user_feat':[8,365],'item_feat':[71, 5, 16, 3]},
    {'dir':'../../../FULL_data/29614120','test_sem':9,'num_crs':1717,'user_feat':[8,365],'item_feat':[85, 9, 36, 7]},
    {'dir':'../../../FULL_data/29614121','test_sem':9,'num_crs':678 ,'user_feat':[8,365],'item_feat':[52, 9, 25, 8]},
    {'dir':'../../../FULL_data/29614122','test_sem':6,'num_crs':98  ,'user_feat':[8,365],'item_feat':[24, 5, 2, 1]},
    {'dir':'../../../FULL_data/29614123','test_sem':9,'num_crs':1177,'user_feat':[8,365],'item_feat':[73, 5, 33, 5]},
    {'dir':'../../../FULL_data/29614124','test_sem':9,'num_crs':774 ,'user_feat':[8,365],'item_feat':[55, 6, 20, 1]},
    {'dir':'../../../FULL_data/29614125','test_sem':9,'num_crs':2355,'user_feat':[8,365],'item_feat':[81, 7, 53, 8]},
    {'dir':'../../../FULL_data/29614126','test_sem':9,'num_crs':332 ,'user_feat':[8,365],'item_feat':[52, 4, 21, 1]},
    {'dir':'../../../FULL_data/29614127','test_sem':9,'num_crs':1190,'user_feat':[8,365],'item_feat':[68, 9, 21, 6]},
]

SYSTEM1_basic_config = {
    'name':'SYSTEM1',
    'load':SYSTEM1_load_data,
    'cuda_num' : Engine.cuda_max_free([0, 1]),
    'course_file' : os.path.join(SYSTEM1_feat_list[SYSTEM1_num]['dir'], 'course.pkl'),
    'schedule' : np.load(os.path.join(SYSTEM1_feat_list[SYSTEM1_num]['dir'], 'schedule.npy')),
    'num_sem' : 6,
    'num_crs' : SYSTEM1_feat_list[SYSTEM1_num]['num_crs'], 
    'test_sem': SYSTEM1_feat_list[SYSTEM1_num]['test_sem'],
    'batch_size' : 64, 
    # ['DEGREE_PURSUED_LEVEL_DESC', 'CIP_MAJOR']
    'stu_features': SYSTEM1_feat_list[SYSTEM1_num]['user_feat'], 
    # ['SYSTEM1 External Subject Area', 'Component', 'Department Code', 'School Code']
    'crs_features' : SYSTEM1_feat_list[SYSTEM1_num]['item_feat'], 
}
    
basic_config = SYSTEM1_basic_config
print(basic_config)

os.environ['CUDA_VISIBLE_DEVICES'] = str(basic_config['cuda_num'])
session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=session_config)

with open(basic_config['course_file'], 'rb') as f:
    stu_dict = pickle.load(f)
    print('Total Number of Students : ' + str(len(stu_dict)))
    
tv_keys, test_keys = basic_config['load'](stu_dict, basic_config['test_sem'])
tv_keys.sort()
train_keys, valid_keys = list_partition(tv_keys, 0.8, seed=0)


importlib.reload(Generator)

num_history = 0
train_generator_config = {
    'name' : None,
    'training' : False, 
    'num_courses_window' : [10, np.inf], 
    'use_same' : False, # X0 == Y.
    'use_sampling' : True, # Sample a number or precentage of courses as input.
    'sampling_level' : 'course',
    'sampling_num_rate' : [0],
    'sampling_window' : [0, basic_config['num_sem']],
    'use_history' : True, # Use first several semester as input.
    'history_num' : basic_config['num_sem'],
    'num_times' : basic_config['num_sem'],
    'predict_future' : False, 
    'batch_size' : 32,
    'shuffle' : True,
    'fixed_seed' : False}

train_generator = Generator.SYSTEM1MultihotGenerator(stu_dict, train_keys, basic_config, train_generator_config)
valid_generator = Generator.SYSTEM1MultihotGenerator(stu_dict, valid_keys, basic_config, train_generator_config)

def generator2feature(generator):
    dataset = generator.__getitem__(batch_index=0, batch_size='MAX')[0]
    course = dataset[0]
    target = dataset[-1]
    return course, target

train_courses, train_target = generator2feature(train_generator)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfPLANBERT
from sklearn.neighbors import NearestNeighbors
import util.Metrics as Metrics

def userKNN(train_courses, test_courses):
    train_stu, train_sem, train_cou = train_courses.shape
    test_stu, test_sem, test_cou = test_courses.shape
    #train_features = (train_courses.sum(1) > 0).astype(float)
    #test_features = (test_courses.sum(1) > 0).astype(float)
    train_features = train_courses.sum(1).astype(float)
    test_features = test_courses.sum(1).astype(float)
    
    train_features += np.random.rand(*train_features.shape) * 1e-10
    test_features += np.random.rand(*test_features.shape) * 1e-10
    sim = cosine_similarity(np.concatenate([train_features, test_features], axis=0))[train_features.shape[0]:, :train_features.shape[0]]
    
    pred = []
    for iter in range(train_sem):
        pred.append(sim.dot(train_courses[:, iter])[:, np.newaxis])
    pred = np.concatenate(pred, 1)
    return pred


test_generator_list = []
test_generator_config = {
    'use_same' : False, # X0 == Y.
    'training' : False, 
    'num_courses_window' : [1, np.inf],
    'use_sampling' : True, # Sample a number or precentage of courses as input.
    'sampling_level' : 'course',
    'sampling_num_rate' : [0],
    'sampling_window' : [0, basic_config['num_sem']],
    'use_history' : True, # Use first several semester as input.
    'history_num' : 0,
    'num_times' : 0,
    'predict_future' : False, 
    'batch_size' : basic_config['batch_size'],
    'shuffle' : True,
    'fixed_seed' : False}
for iter in [0, 2, 4, 5, 6, 8, 10]:
    test_generator_config['sampling_num_rate'] = [iter]
    test_generator_config['history_num'] = 0
    test_generator_config['name'] = '4Y R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
    test_generator = Generator.SYSTEM1MultihotGenerator(stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
    test_generator_list.append(test_generator)
    
for iter in range(0, basic_config['num_sem'], 3):
    test_generator_config['sampling_num_rate'] = [5]
    test_generator_config['history_num'] = iter
    test_generator_config['name'] = '4Y R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
    test_generator = Generator.SYSTEM1MultihotGenerator(stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
    test_generator_list.append(test_generator)

name_list = []
result_list = []
for test_generator in test_generator_list:
    
    test_courses, test_target = generator2feature(test_generator)
    
    user_pred = userKNN(train_courses, test_courses)
    user_pred += np.random.rand(*user_pred.shape) * 1e-10
    recall_mean, recall_per_sem = Metrics.recall(test_target, user_pred, at_n=10)
    
    name_list.append(test_generator.name)
    result_list.append(np.round(recall_mean, 4))
    print('{0} : {1}'.format(test_generator.name, np.round(recall_mean, 4)))
    
import json
save_name = './KNN_result.json'
with open(save_name, 'r') as f:
    result = json.load(f)
    result[SYSTEM1_num] = result_list
with open(save_name, 'w') as f:
    json.dump(result, f)