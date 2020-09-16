import sys
sys.path.append("..")
import os, pickle, importlib, tqdm, sys, keras, copy, Engine
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
import util.Generator as Generator
importlib.reload(Generator)
from util.Datahelper import UNIVERSITY1_load_data, SYSTEM1_load_data, list_partition, list_sampling, list_padding, mat_partition, mat_sampling, mat_padding, set_top_n, list2mat

tf.random.set_random_seed(0)
np.random.seed(0)

UNIVERSITY1_basic_config = {
    'name':'UNIVERSITY1',
    'load':UNIVERSITY1_load_data,
    'cuda_num' : Engine.cuda_max_free([0, 1, 2, 3]),
    'course_file' : '../../../UNIVERSITY1_data/course_data/course.pkl',
    'schedule' : np.load('../../../UNIVERSITY1_data/course_data/schedule.npy'),
    'course_description' : np.load('../../../UNIVERSITY1_data/course_data/course_description.npy'),
    'num_sem' : 12,
    'num_crs' : 2082, 
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
train_keys, valid_keys = list_partition(tv_keys, 1, seed=0)

importlib.reload(Generator)

num_history = 0
train_generator_config = {
    'name' : None,
    'training' : False, 
    'num_courses_window' : [10, np.inf], 
    'stu_type' : 'ALL', 
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

train_generator = Generator.UNIVERSITY1MultihotGenerator(stu_dict, train_keys+valid_keys, basic_config, train_generator_config)

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

test_generator_list = []
for iter in [0, 2, 4, 5, 6, 8, 10]:
    test_generator_config['sampling_num_rate'] = [iter]
    test_generator_config['history_num'] = 0
    test_generator_config['name'] = '4Y R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
    test_generator = Generator.UNIVERSITY1MultihotGenerator(stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
    test_generator_list.append(test_generator)
    
for iter in range(0, basic_config['num_sem'], 3):
    test_generator_config['sampling_num_rate'] = [5]
    test_generator_config['history_num'] = iter
    test_generator_config['name'] = '4Y R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
    test_generator = Generator.UNIVERSITY1MultihotGenerator(stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
    test_generator_list.append(test_generator)
    
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

for iter in range(6, basic_config['num_sem'], 3):
    test_generator_config['sampling_num_rate'] = [3]
    test_generator_config['history_num'] = iter
    test_generator_config['name'] = 'Trans R={0}_H={1}'.format(test_generator_config['sampling_num_rate'], test_generator_config['history_num'])
    test_generator = Generator.UNIVERSITY1MultihotGenerator(stu_dict, test_keys[test_generator_config['history_num']], basic_config, test_generator_config)
    test_generator_list.append(test_generator)
    
def generator2feature(generator):
    dataset = generator.__getitem__(batch_index=0, batch_size='MAX')[0]
    course = dataset[0]
    target = dataset[-1]
    return course, target

train_courses, train_target = generator2feature(train_generator)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
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

for test_generator in test_generator_list:
    print(test_generator.name)
    test_courses, test_target = generator2feature(test_generator)
    
    user_pred = userKNN(train_courses, test_courses)
    user_pred += np.random.rand(*user_pred.shape) * 1e-10
    recall_mean, recall_per_sem = Metrics.recall(test_target, user_pred, at_n=10)
    
    print('User: recall@10 average : {}'.format(np.round(recall_mean, 4)))