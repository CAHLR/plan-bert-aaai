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

UNIVERSITY1_basic_config = {
    'name':'UNIVERSITY1',
    'load':UNIVERSITY1_load_data,
    'cuda_num' : Engine.cuda_max_free([0, 1]),
    'course_file' : '../../../UNIVERSITY1_data/course_data/course.pkl',
    'schedule' : np.load('../../../UNIVERSITY1_data/course_data/schedule.npy'),
    'course_description' : np.load('../../../UNIVERSITY1_data/course_data/course_description.npy'),
    'num_sem' : 12,
    'num_crs' : 7252, 
    'test_sem': 15,
    'batch_size' : 64, 
    # [StudentsCollege, StudentsDivision, StudentsDepartment, StudentsMajor]
    'stu_features': [8, 15, 76, 181],
    # [CoursesDepartment, CoursesSubject, CoursesInstructor, CoursesDescription]
    'crs_features' : [119, 205, 10072, 100]
}
    
basic_config = UNIVERSITY1_basic_config
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

import model.LSTM as LSTM

model_config = {
    'name' : 'LSTM',
    'use_two_direction' : False,
    'num_sem' : basic_config['num_sem'], 
    'num_times' : 0,
    'num_input_0' : [True, basic_config['num_crs'], 'CourseID'],
    'num_input_1' : [True, basic_config['num_sem'], 'RelativeSemester'],
    'num_input_2' : [True, 3, 'Season'],
    'num_input_3' : [True, 1, 'PredictToken'],
    # [whether the feature is used, the dimension of the feature, the name of feature]
    'num_stu_feat_list' : [
        [False, basic_config['stu_features'][0], 'StudentsCollege'],
        [False, basic_config['stu_features'][1], 'StudentsDivision'],
        [False, basic_config['stu_features'][2], 'StudentsDepartment'],
        [False, basic_config['stu_features'][3], 'StudentsMajor']
    ],
    'num_crs_feat_list' : [
        [False, basic_config['crs_features'][0], 'CoursesDepartment'],
        [False, basic_config['crs_features'][1], 'CoursesSubject'],
        [False, basic_config['crs_features'][2], 'CoursesInstructor'],
        [False,basic_config['crs_features'][3], 'CoursesDescription']
    ],
    
    'embedding_dim' : 2**9,
    'num_layers' : 3,
    
    'lstm_dropout' : 0,
    'l2_reg_penalty_weight' : 0,
    'confidence_penalty_weight' : 0,
    'lrate' : 1e-4}

model = LSTM.LSTM(model_config)

# Fine-tune
train_generator_config = {
    'name' : None,
    'training' : True, 
    'stu_type' : 'ALL', 
    'num_courses_window' : [10, np.inf], 
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

train_generator = Generator.UNIVERSITY1MultihotGenerator(stu_dict, train_keys, basic_config, train_generator_config)
valid_generator = Generator.UNIVERSITY1MultihotGenerator(stu_dict, valid_keys, basic_config, train_generator_config)
Engine.fit(
    model=model, 
    train_generator=train_generator, 
    valid_generator=valid_generator, 
    epoch_limit=200, 
    loss_nonimprove_limit=10,
    batch_size=basic_config['batch_size'], 
    use_cosine_lr=True, 
    model_save_path=None)

model.save_weights('./checkpoint/LSTM-10.h5')

result_dict = {'4-Years':{}, 'Trans':{}}

test_generator_config = {
    'use_same' : False, # X0 == Y.
    'training' : False, 
    'stu_type' : 'NEW FRESHMEN', 
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
'''
for h in range(0, basic_config['num_sem'], 3):
    result_dict['4-Years'][h] = {}
    #reference_list = [0, 2, 4, 5, 6, 8, 10] if h == 0 else [5]
    reference_list = [0, 2, 4, 5, 6, 8, 10]
    for r in reference_list:
        test_generator_config['history_num'] = h
        test_generator_config['sampling_num_rate'] = [r]
        
        test_generator_config['name'] = '4-Years R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
        test_generator = Generator.UNIVERSITY1MultihotGenerator(
            stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
        
        print(test_generator.name)
        target_all, predict_all, recall_mean, recall_per_sem = Engine.test(
            model, test_generator, filter='year', history_num=test_generator.history_num, duplicate_filter='pass')
        recall_per_sem = np.round(recall_per_sem, 4)
        recall_mean = np.round(recall_mean, 4)
        print(recall_per_sem)
        print(recall_mean)
        result_dict['4-Years'][h][r] = [recall_mean, list(recall_per_sem)]
'''
for h in range(0, basic_config['num_sem'], 3):
    test_generator_config['history_num'] = h
    test_generator_config['sampling_num_rate'] = [0]
    
    test_generator_config['name'] = '4-Years R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
    test_generator = Generator.UNIVERSITY1MultihotGenerator(
    stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)

    print(test_generator.name)
    target_all, predict_all, recall_mean, recall_per_sem = Engine.test(
    model, test_generator, filter='year', history_num=test_generator.history_num, duplicate_filter='pass')
    recall_per_sem = np.round(recall_per_sem, 4)
    recall_mean = np.round(recall_mean, 4)
    print(recall_per_sem)
    print(recall_mean)
    #result_dict['4-Years'][h][r] = [recall_mean, list(recall_per_sem)]

save_name = './LSTM.json'
with open(save_name, 'w') as f:
    result_dict['Time'] = time.ctime()
    json.dump(result_dict, f)