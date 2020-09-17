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
iter_num = int(sys.argv[2])
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

import model.BERT as BERT
importlib.reload(BERT)

model_config = {
    'name' : 'BERT',
    'mask_future' : False,
    'num_sem' : basic_config['num_sem'], 
    'num_times' : 0,
    'num_input_0' : [True, basic_config['num_crs'], 'CourseID'],
    'num_input_1' : [True, basic_config['num_sem'], 'RelativeSemester'],
    'num_input_2' : [True, 3, 'Season'],
    'num_input_3' : [True, 1, 'PredictToken'],
    
    'embedding_dim' : 2**9,
    'num_layers' : 3,
    'num_heads' : 8,
    
    'PLANBERT_dropout' : 0,
    'embedding_dropout' : 0,
    
    'l2_reg_penalty_weight' : 0,
    'confidence_penalty_weight' : 0.1,
    'lrate' : 1e-4}

model = BERT.BERT(model_config)

# Fine-tune
train_generator_config = {
    'name' : None,
    'training' : True, 
    'num_courses_window' : [1, np.inf], 
    'use_same' : False, # X0 == Y.
    'use_sampling' : True, # Sample a number or precentage of courses as input.
    'sampling_level' : 'course',
    'sampling_num_rate' : [0],
    'sampling_window' : [0, basic_config['num_sem']],
    'use_history' : True, # Use first several semester as input.
    'history_num' : 'random', # ('random', 0, [0,3,6,9])
    'num_times' : 0,
    'predict_future' : False, 
    'batch_size' : basic_config['batch_size'],
    'shuffle' : True,
    'fixed_seed' : False}

train_generator = Generator.SYSTEM1MultihotGenerator(stu_dict, train_keys, basic_config, train_generator_config)
valid_generator = Generator.SYSTEM1MultihotGenerator(stu_dict, valid_keys, basic_config, train_generator_config)
Engine.fit(
    model=model, 
    train_generator=train_generator, 
    valid_generator=valid_generator, 
    epoch_limit=200, 
    loss_nonimprove_limit=5,
    batch_size=basic_config['batch_size'], 
    use_cosine_lr=True, 
    model_save_path=None)

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

result_dict = {0:{}, 3:{}}
for h in range(0, basic_config['num_sem'], 3):
    for r in [0]:
        test_generator_config['sampling_num_rate'] = [r]
        test_generator_config['history_num'] = h
        test_generator_config['name'] = '4Y R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
        test_generator = Generator.SYSTEM1MultihotGenerator(stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
        
        target_all, predict_all, recall_mean, recall_per_sem = Engine.test(model, test_generator, filter='year', history_num=test_generator.history_num)
        result_dict[h][r] = [np.round(recall_mean, 4), [np.round(iter, 4) for iter in recall_per_sem]]
        print('{0} : {1}'.format(test_generator.name, np.round(recall_mean, 4)))
    
import json
save_name = './results/BERT-{0}-{1}.json'.format(SYSTEM1_num, iter_num)
with open(save_name, 'w') as f:
    json.dump(result_dict, f)