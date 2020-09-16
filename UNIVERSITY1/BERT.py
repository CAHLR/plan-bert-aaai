import sys
sys.path.append("..")
import os, pickle, importlib, tqdm, sys, keras, copy, Engine, json, time
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
import util.Generator as Generator
import model.Transformer as Transformer
from util.Datahelper import UNIVERSITY1_load_data, SYSTEM1_load_data, list_partition, list_sampling, list_padding, mat_partition, mat_sampling, mat_padding, set_top_n, list2mat

#id = int(sys.argv[1])
id = 5
num_layers = 3 # int(sys.argv[2])
num_hidden_dims = 9 # int(sys.argv[3])
num_heads = 8 # int(sys.argv[4])
mask_rate = 0.8 # float(sys.argv[5])

use_ref = False
use_item = False
use_user = False
loss_nonimprove_limit = 10

num_ref = 10 if use_ref else 0
model_name = 'BERT' + {True:'+ref', False:''}[use_ref] + {True:'+item', False:''}[use_item] + {True:'+user', False:''}[use_user]
save_name = '{0}-{1}-nl={2}-hd={3}-nh={4}-pt={5}-ft={6}'.format(
    id, model_name, num_layers, num_hidden_dims, num_heads, mask_rate, loss_nonimprove_limit)    

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
    'batch_size' : 32, 
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

model_config = {
    'name' : 'PLAN-BERT',
    'mask_future' : False,
    'num_sem' : basic_config['num_sem'], 
    'num_times' : 0,
    'num_input_0' : [True, basic_config['num_crs'], 'CourseID'],
    'num_input_1' : [True, basic_config['num_sem'], 'RelativeSemester'],
    'num_input_2' : [True, 3, 'Season'],
    'num_input_3' : [True, 1, 'PredictToken'],
    # [whether the feature is used, the dimension of the feature, the name of feature]
    'num_stu_feat_list' : [
        [use_user, basic_config['stu_features'][0], 'StudentsCollege'],
        [use_user, basic_config['stu_features'][1], 'StudentsDivision'],
        [use_user, basic_config['stu_features'][2], 'StudentsDepartment'],
        [use_user, basic_config['stu_features'][3], 'StudentsMajor']
    ],
    'num_crs_feat_list' : [
        [use_item, basic_config['crs_features'][0], 'CoursesDepartment'],
        [use_item, basic_config['crs_features'][1], 'CoursesSubject'],
        [False, basic_config['crs_features'][2], 'CoursesInstructor'],
        [False,basic_config['crs_features'][3], 'CoursesDescription']
    ],
    
    'embedding_dim' : 2**num_hidden_dims,
    'num_layers' : num_layers,
    'num_heads' : 8,
    
    'transformer_dropout' : 0.2,
    'embedding_dropout' : 0.2,
    
    'l2_reg_penalty_weight' : 0,
    'confidence_penalty_weight' : 0.1,
    'lrate' : 1e-4}

model = Transformer.Transformer(model_config)

if use_ref:
    # Pretraining : Course-level Masking
    train_generator_config = {
        'name' : None,
        'training' : True, 
        'stu_type' : 'ALL', 
        'num_courses_window' : [10, np.inf], 
        'use_same' : False, # X0 == Y.
        'use_sampling' : True, # Sample a number or precentage of courses as input.
        'sampling_level' : 'course',
        'sampling_num_rate' : [mask_rate],
        'sampling_window' : [0, basic_config['num_sem']],
        'use_history' : False, # Use first several semester as input.
        'history_num' : 0,
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
        loss_nonimprove_limit=loss_nonimprove_limit,
        batch_size=basic_config['batch_size'], 
        use_cosine_lr=True, 
        model_save_path=None)

# Fine-tune
train_generator_config = {
    'name' : None,
    'training' : True, 
    'stu_type' : 'ALL', 
    'num_courses_window' : [10, np.inf], 
    'use_same' : False, # X0 == Y.
    'use_sampling' : True, # Sample a number or precentage of courses as input.
    'sampling_level' : 'course',
    'sampling_num_rate' : [num_ref],
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
    loss_nonimprove_limit=loss_nonimprove_limit,
    batch_size=basic_config['batch_size'], 
    use_cosine_lr=True, 
    model_save_path=None)

model.save_weights('../checkpoint/{0}.h5'.format(save_name))

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

for h in range(0, basic_config['num_sem'], 3):
    result_dict['4-Years'][h] = {}
    for r in [0]:
        test_generator_config['history_num'] = h
        test_generator_config['sampling_num_rate'] = [r]
        
        test_generator_config['name'] = '4-Years R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
        test_generator = Generator.UNIVERSITY1MultihotGenerator(
            stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
        
        print(test_generator.name)
        target_all, predict_all, recall_mean, recall_per_sem = Engine.test(
            model, test_generator, filter='year', history_num=test_generator.history_num)
        recall_per_sem = np.round(recall_per_sem, 4)
        recall_mean = np.round(recall_mean, 4)
        print(recall_per_sem)
        print(recall_mean)
        result_dict['4-Years'][h][r] = [recall_mean, list(recall_per_sem)]

test_generator_config = {
    'use_same' : False, # X0 == Y.
    'training' : False, 
    'stu_type' : 'ADVANCED STANDING', 
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

for h in range(6, basic_config['num_sem'], 3):
    result_dict['Trans'][h] = {}
    for r in [0]:
        test_generator_config['history_num'] = h
        test_generator_config['sampling_num_rate'] = [r]
        
        test_generator_config['name'] = 'Trans R={0}_H={1}'.format(
            test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
        test_generator = Generator.UNIVERSITY1MultihotGenerator(
            stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
        
        print(test_generator.name)
        target_all, predict_all, recall_mean, recall_per_sem = Engine.test(
            model, test_generator, filter='year', history_num=test_generator.history_num)
        recall_per_sem = np.round(recall_per_sem, 4)
        recall_mean = np.round(recall_mean, 4)
        print(recall_per_sem)
        print(recall_mean)
        result_dict['Trans'][h][r] = [recall_mean, list(recall_per_sem)]

with open(save_name + '.json', 'w') as f:
    result_dict['Time'] = time.ctime()
    json.dump(result_dict, f)