import os, pickle, tqdm, keras, time
import numpy as np
np.random.seed(0)
from copy import deepcopy
from util.Datahelper import list_partition, list_sampling, list_padding, mat_partition, mat_sampling, mat_padding, set_top_n, list2mat

num_courses = 2830
num_majors = 266
num_semesters = 12
token_dict = {
    '[PADDING]':0 + num_courses, 
    '[MASK]':1 + num_courses, 
    '[PREDICTION]':2 + num_courses, 
    '[SEP]':3 + num_courses}


def RoBERTa_masking(List, num_courses):
    '''
    Dynamic Masking from 'RoBERTa: A Robustly Optimized BERT Pretraining Approach'
    15% of possible replacement.
    
    80% use [MASK] token.
    10% are unchanged.
    10% are randomly changed.
    '''
    
    List_2 = deepcopy(List)
    random_list = np.random.rand(len(List), 2)
    replace_courses = np.random.randint(low=0, high=num_courses, size=[len(List)])
    
    possible_replacement_index = (random_list[:, 0] < 0.15) * (List_2 < num_courses) # Never replace tokens.
    mask_index = possible_replacement_index * (random_list[:, 1] < 0.8)
    unchange_index = possible_replacement_index * (random_list[:, 1] > 0.8) * (random_list[:, 1] < 0.9)
    replace_index = possible_replacement_index * (random_list[:, 1] > 0.9)
    
    List_2[mask_index] = token_dict['[MASK]']
    List_2[replace_index] = replace_courses[replace_index]
    return List_2, possible_replacement_index


def onehot_encoding(List, shape):
    List_2 = np.zeros(shape)
    List_2[np.arange(len(List)).astype(int), np.array(List).astype(int)] = 1
    return List_2

def multihot_encoding(List, shape):
    List_2 = np.zeros(shape)
    for x, iter in enumerate(List):
        List_2[x, iter] = 1
    return List_2



class SYSTEM1MultihotGenerator(keras.utils.Sequence):
    def __init__(self, data, stu_ids, basic_config, generator_config):
        super(SYSTEM1MultihotGenerator, self).__init__()
        
        self.name = generator_config['name']
        self.data = data
        self.stu_ids = stu_ids
        print('Length of dataset : {}'.format(len(self.stu_ids)))
        
        self.basic_config = basic_config
        self.generator_config = generator_config
        
        self.schedule = basic_config['schedule']
        
        self.num_crs = self.basic_config['num_crs']
        self.num_sem = self.basic_config['num_sem']
        self.stu_feat = basic_config['stu_features']
        self.crs_feat = basic_config['crs_features']
        
        data2 = {}
        if 'num_courses_window' in generator_config:
            for each in data:
                data[each]['item_feats'] = data[each]['item_feats'][data[each]['item_feats'][:, 0] < self.num_sem]
                if (data[each]['item_feats'].shape[0] >= generator_config['num_courses_window'][0]) and (data[each]['item_feats'].shape[0] < generator_config['num_courses_window'][1]):
                    data2[each] = data[each]
            self.data = data2
        self.stu_ids = [iter for iter in filter(lambda x:(x in self.data), self.stu_ids)]
        print('Length of dataset : {}'.format(len(self.stu_ids)))
        
        self.training = self.generator_config['training'] # If True, return schedule for each semester.
        self.use_same = self.generator_config['use_same'] # If True, input and output courses are the same.
        self.use_sampling = self.generator_config['use_sampling']
        
        self.sampling_level = self.generator_config['sampling_level']
        assert(self.sampling_level in ['course', 'semester'])
        
        '''
        If < 1, do percentage sampling.
        If >=1, do number sampling.
        '''
        self.sampling_num_rate = self.generator_config['sampling_num_rate']
        self.sampling_window = self.generator_config['sampling_window']
        self.use_history = self.generator_config['use_history']
        self.history_num = self.generator_config['history_num']
        self.num_times = self.generator_config['num_times']
        self.predict_future = self.generator_config['predict_future']
        
        self.batch_size = self.generator_config['batch_size']
        self.shuffle = self.generator_config['shuffle']
        self.fixed_seed = self.generator_config['fixed_seed']
        
        self.num_epoch = 0
        self.on_epoch_end()
        
        
    def __len__(self):
        if (len(self.stu_ids) % self.batch_size) == 0:
            return int(len(self.stu_ids)//self.batch_size)
        else:
            return int(len(self.stu_ids)//self.batch_size + 1)
    
    
    def __getitem__(self, batch_index, batch_size=None):
        if batch_size == None:
            stu_indexes = self.stu_ids[(batch_index * self.batch_size):((batch_index + 1) * self.batch_size)]
        elif batch_size == 'MAX':
            stu_indexes = self.stu_ids
        else:
            stu_indexes = self.stu_ids[(batch_index * batch_size):((batch_index + 1) * batch_size)]
        X0, X1, X2, X3, X4, X5, X6, X7, X8, X9 = [], [], [], [], [], [], [], [], [], []
        S = [[], []]
        Y = []
        
        for stu_index in stu_indexes:
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, y = self.data_generation(stu_index)
            X0.append(x0[np.newaxis, :, :])
            X1.append(x1[np.newaxis, :, :])
            X2.append(x2[np.newaxis, :, :])
            X3.append(x3[np.newaxis, :, :])
            X4.append(x4[np.newaxis, :, :])
            X5.append(x5[np.newaxis, :, :])
            X6.append(x6[np.newaxis, :, :])
            X7.append(x7[np.newaxis, :, :])
            X8.append(x8[np.newaxis, :, :])
            X9.append(x9[np.newaxis, :, :])
            Y.append(y[np.newaxis, :, :])
            
            if not self.training:
                newest_sem = int(self.data[stu_index]['start'] + self.history_num) if self.use_history else int(self.data[stu_index]['start'])
                S[0].append(np.tile(self.schedule[np.newaxis, newest_sem-3:newest_sem], [1, self.num_sem//3, 1])) # Year Filter
                S[1].append(np.tile(self.schedule[np.newaxis, :newest_sem].sum(1) > 0, [1, self.num_sem, 1])) # Union Filter

        X0 = np.concatenate(X0, axis=0)
        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        X3 = np.concatenate(X3, axis=0)
        X4 = np.concatenate(X4, axis=0)
        X5 = np.concatenate(X5, axis=0)
        X6 = np.concatenate(X6, axis=0)
        X7 = np.concatenate(X7, axis=0)
        X8 = np.concatenate(X8, axis=0)
        X9 = np.concatenate(X9, axis=0)
        Y = np.concatenate(Y, axis=0)
        
        if not self.training:
            S[0] = np.concatenate(S[0], axis=0)
            S[1] = np.concatenate(S[1], axis=0)
            return [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, Y], S
        else:
            return [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, Y], []
    
    
    def on_epoch_end(self):
        if self.shuffle == True:
            self.num_epoch += 1
            # Shuffle the order of inputs.
            np.random.seed(self.num_epoch)
            np.random.shuffle(self.stu_ids)
    
    
    def data_generation(self, index):
        # If use fixed seed, use the ppsk of students as the seed of sampling.
        temp_seed = int(index) if self.fixed_seed else (int(index) + self.num_epoch)
        temp_start = self.data[index]['start']
        
        temp_courses = self.data[index]['item_feats']
        temp_majors = self.data[index]['user_feats']

        temp_courses = temp_courses[(temp_courses[:, 0] >= 0) * (temp_courses[:, 0] < self.num_sem), :]
        temp_majors = temp_majors[(temp_majors[:, 0] >= 0) * (temp_majors[:, 0] < self.num_sem), :]
        
        known_crs_list = []
        unknown_crs_list = []
        
        if self.use_same:
            pass
        else:
            if self.use_sampling:
                temp_sampling_rate = self.sampling_num_rate[np.random.randint(low=0, high=len(self.sampling_num_rate))]
                if temp_sampling_rate >= 1:
                    temp_sample_num = int(np.round(temp_sampling_rate))
                else:
                    assert temp_sampling_rate >= 0 # 0 < rate < 1
                    temp_sample_num = int(np.round(temp_sampling_rate * temp_courses.shape[0]))
                    
                sample_courses, remaining_courses = list_sampling(
                    temp_courses, temp_sample_num, self.sampling_window, seed=temp_seed)
                known_crs_list += list(sample_courses)
                unknown_crs_list += list(remaining_courses)
            
            if self.use_history:
                if self.history_num == 'random':
                    temp_history_num = np.random.randint(low=0, high=self.num_sem)
                elif type(self.history_num) == list:
                    temp_history_num = self.history_num[np.random.randint(low=0, high=len(self.history_num))]
                else:
                    temp_history_num = self.history_num
                history_index = (remaining_courses[:, 0] < temp_history_num)
                known_crs_list += list(remaining_courses[np.where(history_index)[0]])
                unknown_crs_list = list(remaining_courses[np.where(1 - history_index)[0]])
            else:
                temp_history_num = 0
            
            known_crs_list = np.array(known_crs_list).reshape([-1, 6])
            unknown_crs_list = np.array(unknown_crs_list).reshape([-1, 6])
            known_stu_list = temp_majors[(temp_majors[:, 0] < temp_history_num + 1), :].reshape([-1, 3])
            
            # CourseID
            x0 = list2mat(known_crs_list[:, [0, 1]], [self.num_sem, self.num_crs])
            # Relative Semester
            x1 = np.eye(self.num_sem)
            # Season Embedding
            x2 = onehot_encoding((np.arange(self.num_sem) + temp_start) % 3, [self.num_sem, 3])
            # Predict Token
            x3 = np.concatenate([np.zeros([temp_history_num, 1]), np.ones([self.num_sem-temp_history_num, 1])], axis=0)
            
            # Student's Pursued Degree Embedding
            x4 = list2mat(known_stu_list[:, [0, 1]], [self.num_sem, self.stu_feat[0]])
            # Student's Major Embedding
            x5 = list2mat(known_stu_list[:, [0, 2]], [self.num_sem, self.stu_feat[1]])
            
            # Courses' Subjects Embedding
            x6 = list2mat(known_crs_list[:, [0, 2]], [self.num_sem, self.crs_feat[0]])
            # Courses' Components Embedding
            x7 = list2mat(known_crs_list[:, [0, 3]], [self.num_sem, self.crs_feat[1]])
            # Courses' Department Embedding
            x8 = list2mat(known_crs_list[:, [0, 4]], [self.num_sem, self.crs_feat[2]])
            # Courses' School Embedding
            x9 = list2mat(known_crs_list[:, [0, 5]], [self.num_sem, self.crs_feat[3]])
            
            # Target
            y = list2mat(unknown_crs_list[:, [0, 1]], [self.num_sem, self.num_crs])
            return x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, y




class UNIVERSITY1MultihotGenerator(keras.utils.Sequence):
    def __init__(self, data, stu_ids, basic_config, generator_config):
        super(UNIVERSITY1MultihotGenerator, self).__init__()
        
        self.name = generator_config['name']
        self.data = data
        self.stu_ids = stu_ids
        print('Length of dataset : {}'.format(len(self.stu_ids)))
        
        self.basic_config = basic_config
        self.generator_config = generator_config
        
        self.schedule = basic_config['schedule']
        self.description = basic_config['course_description']
        
        self.num_crs = self.basic_config['num_crs']
        self.num_sem = self.basic_config['num_sem']
        self.stu_feat = basic_config['stu_features']
        self.crs_feat = basic_config['crs_features']
        
        self.training = self.generator_config['training'] # If True, return schedule for each semester.
        self.use_same = self.generator_config['use_same'] # If True, input and output courses are the same.
        self.use_sampling = self.generator_config['use_sampling']
        
        
        
        self.sampling_level = self.generator_config['sampling_level']
        assert(self.sampling_level in ['course', 'semester'])
        
        '''
        If < 1, do percentage sampling.
        If >=1, do number sampling.
        '''
        self.sampling_num_rate = self.generator_config['sampling_num_rate']
        self.sampling_window = self.generator_config['sampling_window']
        self.use_history = self.generator_config['use_history']
        self.history_num = self.generator_config['history_num']
        self.num_times = self.generator_config['num_times']
        self.predict_future = self.generator_config['predict_future']
        
        if max(self.sampling_num_rate) >= 1:
            generator_config['num_courses_window'][0] = max(self.sampling_num_rate) + 1
        self.min_crs = generator_config['num_courses_window'][0]
        self.max_crs = generator_config['num_courses_window'][1]
        
        stu_ids2 = []
        for each in stu_ids:
            if (self.generator_config['stu_type']=='ALL') or (data[each]['type']==self.generator_config['stu_type']):
                if 'num_courses_window' in generator_config:
                    if (self.data[each]['courses'].shape[0] >= self.min_crs) and (self.data[each]['courses'].shape[0] < self.max_crs):
                        stu_ids2.append(each)
        self.stu_ids = [iter for iter in filter(lambda x:(x in self.data), stu_ids2)]
        print('Length of dataset : {}'.format(len(self.stu_ids)))
        
        self.batch_size = self.generator_config['batch_size']
        self.shuffle = self.generator_config['shuffle']
        self.fixed_seed = self.generator_config['fixed_seed']
        
        self.num_epoch = 0
        self.on_epoch_end()
        
        
    def __len__(self):
        return int(len(self.stu_ids) / self.batch_size)
    
    
    def __getitem__(self, batch_index, batch_size=None):
        if batch_size == None:
            stu_indexes = self.stu_ids[(batch_index * self.batch_size):((batch_index + 1) * self.batch_size)]
        elif batch_size == 'MAX':
            stu_indexes = self.stu_ids
        else:
            stu_indexes = self.stu_ids[(batch_index * batch_size):((batch_index + 1) * batch_size)]
        X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11 = [], [], [], [], [], [], [], [], [], [], [], []
        S = [[], []]
        Y = []
        
        for stu_index in stu_indexes:
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y = self.data_generation(stu_index)
            X0.append(x0[np.newaxis, :, :])
            X1.append(x1[np.newaxis, :, :])
            X2.append(x2[np.newaxis, :, :])
            X3.append(x3[np.newaxis, :, :])
            X4.append(x4[np.newaxis, :, :])
            X5.append(x5[np.newaxis, :, :])
            X6.append(x6[np.newaxis, :, :])
            X7.append(x7[np.newaxis, :, :])
            X8.append(x8[np.newaxis, :, :])
            X9.append(x9[np.newaxis, :, :])
            X10.append(x10[np.newaxis, :, :])
            X11.append(x11[np.newaxis, :, :])
            Y.append(y[np.newaxis, :, :])
            
            if not self.training:
                newest_sem = int(self.data[stu_index]['start'] + self.history_num) if self.use_history else int(self.data[stu_index]['start'])
                S[0].append(np.tile(self.schedule[np.newaxis, newest_sem-3:newest_sem], [1, self.num_sem//3, 1])) # Year Filter
                S[1].append(np.tile(self.schedule[np.newaxis, :newest_sem].sum(1) > 0, [1, self.num_sem, 1])) # Union Filter

        X0 = np.concatenate(X0, axis=0)
        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        X3 = np.concatenate(X3, axis=0)
        X4 = np.concatenate(X4, axis=0)
        X5 = np.concatenate(X5, axis=0)
        X6 = np.concatenate(X6, axis=0)
        X7 = np.concatenate(X7, axis=0)
        X8 = np.concatenate(X8, axis=0)
        X9 = np.concatenate(X9, axis=0)
        X10 = np.concatenate(X10, axis=0)
        X11 = np.concatenate(X11, axis=0)
        Y = np.concatenate(Y, axis=0)
        
        if not self.training:
            S[0] = np.concatenate(S[0], axis=0)
            S[1] = np.concatenate(S[1], axis=0)
            return [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, Y], S
        else:
            return [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, Y], []
    
    
    def on_epoch_end(self):
        if self.shuffle == True:
            self.num_epoch += 1
            # Shuffle the order of inputs.
            np.random.seed(self.num_epoch)
            np.random.shuffle(self.stu_ids)
    
    
    def data_generation(self, index):
        # If use fixed seed, use the ppsk of students as the seed of sampling.
        temp_seed = int(index) if self.fixed_seed else (int(index) + self.num_epoch)
        temp_start = self.data[index]['start']
        
        temp_courses = np.array(self.data[index]['courses'])
        temp_majors = np.array(self.data[index]['majors'])
        
        known_ids_list = []
        unknown_ids_list = []
        
        if self.use_same:
            pass
        else:
            if self.use_sampling:
                temp_sampling_rate = self.sampling_num_rate[np.random.randint(low=0, high=len(self.sampling_num_rate))]
                if temp_sampling_rate >= 1:
                    temp_sample_num = int(temp_sampling_rate)
                else:
                    assert temp_sampling_rate >= 0 # 0 < rate < 1
                    temp_sample_num = int(temp_sampling_rate * temp_courses.shape[0])
                
                sample_ids, remaining_ids = list_sampling(
                    np.arange(temp_courses.shape[0])[:, np.newaxis], temp_sample_num, None, seed=temp_seed)
                known_ids_list += list(sample_ids[:, 0])
            
            if self.use_history:
                if self.history_num == 'random':
                    temp_history_num = np.random.randint(low=0, high=self.num_sem)
                elif type(self.history_num) == list:
                    temp_history_num = self.history_num[np.random.randint(low=0, high=len(self.history_num))]
                else:
                    temp_history_num = self.history_num
                known_ids_list += list(np.where(temp_courses[:, 0] < temp_history_num)[0])
                
            else:
                temp_history_num = 0
            
            known_ids_list = np.unique(known_ids_list)
            unknown_ids_list = list(filter(lambda x:x not in known_ids_list, np.arange(temp_courses.shape[0])))
            
            known_crs_list = temp_courses[known_ids_list] if len(known_ids_list) > 0 else np.zeros([0, 5])
            unknown_crs_list = temp_courses[unknown_ids_list] if len(unknown_ids_list) > 0 else np.zeros([0, 5])
            known_user_feat_list = temp_majors[(temp_majors[:, 0] < temp_history_num + 1), :]
            
            # CourseID
            x0 = list2mat(known_crs_list[:, [0, 1]], [self.num_sem, self.num_crs])
            # Relative Semester
            x1 = np.eye(self.num_sem)
            # Season Embedding
            x2 = onehot_encoding((np.arange(self.num_sem) + temp_start) % 3, [self.num_sem, 3])
            # Predict Token
            x3 = np.concatenate([np.zeros([temp_history_num, 1]), np.ones([self.num_sem-temp_history_num, 1])], axis=0)
            
            # Students' College Embedding
            x4 = list2mat(known_user_feat_list[:, [0, 1]], [self.num_sem, self.stu_feat[0]])
            # Students' Division Embedding
            x5 = list2mat(known_user_feat_list[:, [0, 2]], [self.num_sem, self.stu_feat[1]])
            # Students' Department Embedding
            x6 = list2mat(known_user_feat_list[:, [0, 3]], [self.num_sem, self.stu_feat[2]])
            # Students' Major Embedding
            x7 = list2mat(known_user_feat_list[:, [0, 4]], [self.num_sem, self.stu_feat[3]])
            
            # Courses' Department Embedding
            x8 = list2mat(known_crs_list[:, [0, 2]], [self.num_sem, self.crs_feat[0]])
            # Courses' Subject Embedding
            x9 = list2mat(known_crs_list[:, [0, 3]], [self.num_sem, self.crs_feat[1]])
            # Courses' Instructor Embedding
            x10 = np.zeros([self.num_sem, self.crs_feat[2]])
            for known_crs in known_crs_list:
                x10[int(known_crs[0]), [int(each) for each in known_crs[4]]] = 1
            # Courses' Description Embedding
            x11 = np.zeros([self.num_sem, self.crs_feat[3]])
            #for known_crs in known_crs_list:
            #    x6[int(known_crs[0])] += self.description[known_crs[1].astype(int)]
            
            # Target
            y = list2mat(unknown_crs_list[:, [0, 1]], [self.num_sem, self.num_crs])
            
            return x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y

    
class OnehotGenerator(keras.utils.Sequence):
    def __init__(self, data, stu_ids, basic_config, generator_config):
        super(OnehotGenerator, self).__init__()
        
        self.data = data
        self.stu_ids = stu_ids
        self.name = generator_config['name']
        
        self.basic_config = basic_config
        self.generator_config = generator_config
        
        self.num_semesters = self.basic_config['num_semesters']
        self.num_features = [30, 2082, 119, 205, 10072, 100, 8, 15, 76, 181]
        
        data2 = {}
        if 'num_courses_window' in generator_config:
            for each in data:
                data[each]['courses'] = data[each]['courses'][data[each]['courses'][:, 0] < self.num_semesters]
                if (data[each]['courses'].shape[0] >= generator_config['num_courses_window'][0]) and (data[each]['courses'].shape[0] < generator_config['num_courses_window'][1]):
                    if (self.generator_config['stu_type']=='ALL') or (data[each]['type']==self.generator_config['stu_type']):
                        data2[each] = data[each]
            self.data = data2
        self.stu_ids = [iter for iter in filter(lambda x:(x in self.data), self.stu_ids)]
        print('Length of dataset : {}'.format(len(self.stu_ids)))
        
        self.use_same = self.generator_config['use_same']
        self.use_sampling = self.generator_config['use_sampling']
        self.sampling_num_rate = self.generator_config['sampling_num_rate']
        self.sampling_window = self.generator_config['sampling_window']
        self.use_history = self.generator_config['use_history']
        self.history_num = self.generator_config['history_num']
        self.num_times = self.generator_config['num_times']
        self.batch_size = self.generator_config['batch_size']
        self.shuffle = self.generator_config['shuffle']
        self.fixed_seed = self.generator_config['fixed_seed']
        
        self.training = self.generator_config['training'] # If True, return schedule for each semester.
        self.schedule = basic_config['schedule']
        self.description = basic_config['course_description']
        
        self.batch_size = self.generator_config['batch_size']
        self.shuffle = self.generator_config['shuffle']
        self.fixed_seed = self.generator_config['fixed_seed']
        
        self.num_epoch = 0
        self.on_epoch_end()
        
        
    def __len__(self):
        return int(len(self.stu_ids) / self.batch_size)
    
    
    def __getitem__(self, batch_index, batch_size=None):
        if batch_size == None:
            stu_indexes = self.stu_ids[(batch_index * self.batch_size):((batch_index + 1) * self.batch_size)]
        else:
            stu_indexes = self.stu_ids[(batch_index * batch_size):((batch_index + 1) * batch_size)]
        X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11 = [], [], [], [], [], [], [], [], [], [], [], []
        S = [[], []]
        P = []
        Y = []
        
        for stu_index in stu_indexes:
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y, p = self.__data_generation(stu_index)
            X0.append(x0[np.newaxis, :, :])
            X1.append(x1[np.newaxis, :, :])
            X2.append(x2[np.newaxis, :, :])
            X3.append(x3[np.newaxis, :, :])
            X4.append(x4[np.newaxis, :, :])
            X5.append(x5[np.newaxis, :, :])
            X6.append(x6[np.newaxis, :, :])
            X7.append(x7[np.newaxis, :, :])
            X8.append(x8[np.newaxis, :, :])
            X9.append(x9[np.newaxis, :, :])
            X10.append(x10[np.newaxis, :, :])
            X11.append(x11[np.newaxis, :, :])
            Y.append(y[np.newaxis, :, :])
            P.append(p[np.newaxis, :])
            
            if not self.training:
                newest_sem = int(self.data[stu_index]['start'] + self.history_num) if self.use_history else int(self.data[stu_index]['start'])
                S[0].append(np.tile(self.schedule[np.newaxis, newest_sem-3:newest_sem], [1, self.num_semesters//3, 1])) # Year Filter
                S[1].append(np.tile(self.schedule[np.newaxis, :newest_sem].sum(1) > 0, [1, self.num_semesters, 1])) # Union Filter

        X0 = np.concatenate(X0, axis=0)
        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        X3 = np.concatenate(X3, axis=0)
        X4 = np.concatenate(X4, axis=0)
        X5 = np.concatenate(X5, axis=0)
        X6 = np.concatenate(X6, axis=0)
        X7 = np.concatenate(X7, axis=0)
        X8 = np.concatenate(X8, axis=0)
        X9 = np.concatenate(X9, axis=0)
        X10 = np.concatenate(X10, axis=0)
        X11 = np.concatenate(X11, axis=0)
        Y = np.concatenate(Y, axis=0)
        P = np.concatenate(P, axis=0)
        
        if not self.training:
            S[0] = np.concatenate(S[0], axis=0)
            S[1] = np.concatenate(S[1], axis=0)
            return [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, Y, P], S
        else:
            return [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, Y, P], []
    
    
    def on_epoch_end(self):
        if self.shuffle == True:
            self.num_epoch += 1
            # Shuffle the order of inputs.
            np.random.seed(self.num_epoch)
            np.random.shuffle(self.stu_ids)
    
    
    def __data_generation(self, index):
        # If use fixed seed, use the ppsk of students as the seed of sampling.
        temp_seed = index if self.fixed_seed else (index + self.num_epoch)
        temp_start = self.data[index]['start']
        
        temp_courses = self.data[index]['courses']#.astype(int)
        temp_majors = self.data[index]['majors']#.astype(int)
        if self.data[index]['type'] == 'ADVANCED STANDING':
            # Add padding of 6 semester for transfer students.
            temp_courses = temp_courses[(temp_courses[:, 0] >= 0) * (temp_courses[:, 0] < self.num_semesters-6), :]
            temp_majors = temp_majors[(temp_majors[:, 0] >= 0) * (temp_majors[:, 0] < self.num_semesters-6), :]
            temp_start += 6
            temp_courses[:, 0] += 6
            temp_majors[:, 0] += 6
        else:
            temp_courses = temp_courses[(temp_courses[:, 0] >= 0) * (temp_courses[:, 0] < self.num_semesters), :]
            temp_majors = temp_majors[(temp_majors[:, 0] >= 0) * (temp_majors[:, 0] < self.num_semesters), :]
        
        known_courses = []
        unknown_courses = []
        known_majors = []
        p = np.array([1]*self.num_semesters + [0]*(self.num_times))
        
        if self.use_same:
            pass
        else:
            if self.use_sampling:
                temp_sampling_rate = self.sampling_num_rate[np.random.randint(low=0, high=len(self.sampling_num_rate))]
                if temp_sampling_rate >= 1:
                    temp_sample_num = int(np.round(temp_sampling_rate))
                else:
                    assert temp_sampling_rate >= 0 # 0 < rate < 1
                    temp_sample_num = int(np.round(temp_sampling_rate * temp_courses.shape[0]))
                    
                sample_courses, remaining_courses = list_sampling(
                    temp_courses, temp_sample_num, self.sampling_window, seed=temp_seed)
                known_courses += list(sample_courses)
                unknown_courses += list(remaining_courses)
                known_majors = np.arange(1)
            
            if self.use_history:
                if self.history_num == 'random':
                    temp_history_num = np.random.randint(low=0, high=self.num_semesters)
                elif type(self.history_num) == list:
                    temp_history_num = self.history_num[np.random.randint(low=0, high=len(self.history_num))]
                else:
                    temp_history_num = self.history_num
                history_index = (remaining_courses[:, 0] < temp_history_num)
                known_courses += list(remaining_courses[np.where(history_index)[0]])
                unknown_courses = list(remaining_courses[np.where(1 - history_index)[0]])
                known_majors = np.arange(temp_history_num + 1)
                p[:temp_history_num] = 0
            else:
                temp_history_num = 0
            
            known_courses = np.array(known_courses).reshape([-1, 5])
            unknown_courses = np.array(unknown_courses).reshape([-1, 5])
            known_majors = temp_majors[(temp_majors[:, 0] < temp_history_num + 1), :].reshape([-1, 5])
            
            # CourseID
            x0 = np.concatenate([
                list2mat(known_courses[:, [0, 1]], [self.num_semesters, self.num_features[1]]),
                #np.zeros([self.num_semesters, self.num_features[1]]),
                onehot_encoding(known_courses[:, 1], [self.num_times, self.num_features[1]])
            ], axis=0)
            # Relative Semester
            x1 = np.concatenate([
                np.eye(self.num_semesters),
                onehot_encoding(known_courses[:, 0], [self.num_times, self.num_semesters])
            ], axis=0)
            # Season Embedding
            x2 = np.concatenate([
                onehot_encoding((np.arange(self.num_semesters) + temp_start) % 3, [self.num_semesters, 3]),
                onehot_encoding((known_courses[:, 0] + temp_start) % 3, [self.num_times, 3])
            ], axis=0)
            # Courses' Department Embedding
            x3 = np.concatenate([
                list2mat(known_courses[:, [0, 2]], [self.num_semesters, self.num_features[2]]), 
                #np.zeros([self.num_semesters, self.num_features[2]]),
                onehot_encoding(known_courses[:, 2], [self.num_times, self.num_features[2]])
            ], axis=0)
            # Courses' Subject Embedding
            x4 = np.concatenate([
                list2mat(known_courses[:, [0, 3]], [self.num_semesters, self.num_features[3]]),
                #np.zeros([self.num_semesters, self.num_features[3]]),
                onehot_encoding(known_courses[:, 3], [self.num_times, self.num_features[3]])
            ], axis=0)
            # Courses' Instructor Embedding
            x5 = np.zeros([self.num_semesters, self.num_features[4]])
            for known_course in known_courses:
                x5[int(known_course[0]), [int(each) for each in known_course[4]]] = 1
            x5 = np.concatenate([
                x5,
                multihot_encoding(known_courses[:, 4], [self.num_times, self.num_features[4]])
            ], axis=0)
            # Courses' Description Embedding
            x6 = np.concatenate([
                np.zeros([self.num_semesters, self.num_features[5]]),
                self.description[known_courses[:, 1].astype(int)],
                np.zeros([self.num_times-len(known_courses[:, 1]), self.num_features[5]]),
            ], axis=0)
            # Students' College Embedding
            x7 = np.concatenate([
                list2mat(known_majors[:, [0, 1]], [self.num_semesters, self.num_features[6]]),
                np.zeros([self.num_times, self.num_features[6]])
            ], axis=0)
            # Students' Division Embedding
            x8 = np.concatenate([
                list2mat(known_majors[:, [0, 2]], [self.num_semesters, self.num_features[7]]),
                np.zeros([self.num_times, self.num_features[7]])
            ], axis=0)
            # Students' Department Embedding
            x9 = np.concatenate([
                list2mat(known_majors[:, [0, 3]], [self.num_semesters, self.num_features[8]]),
                np.zeros([self.num_times, self.num_features[8]])
            ], axis=0)
            # Students' Major Embedding
            x10 = np.concatenate([
                list2mat(known_majors[:, [0, 4]], [self.num_semesters, self.num_features[9]]),
                np.zeros([self.num_times, self.num_features[9]])
            ], axis=0)
            # Predict Token
            x11 = np.concatenate([
                np.zeros([temp_history_num, 1]),
                np.ones([self.num_semesters-temp_history_num, 1]),
                np.zeros([self.num_times, 1])
            ], axis=0)
            # Target
            y = np.concatenate([
                list2mat(unknown_courses[:, [0, 1]], [self.num_semesters, self.num_features[1]]), 
                np.zeros([self.num_times, self.num_features[1]])], 
                axis=0)
            return x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y, p
        
        
class MaskedLanguageModelGenerator(keras.utils.Sequence):
    def __init__(self, data, stu_ids, mask_rate, 
                 num_semesters, num_courses, num_majors, 
                 batch_size=32, shuffle=True, fixed_seed=False):
        super(MaskedLanguageModelGenerator, self).__init__()
        
        self.data = data
        self.stu_ids = stu_ids
        
        self.num_semesters = num_semesters
        self.num_courses = num_courses
        self.num_majors = num_majors
        
        self.mask_rate = mask_rate
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.fixed_seed = fixed_seed
        self.num_epoch = 0
        
        self.on_epoch_end()
        
        
    def __len__(self):
        return int(len(self.stu_ids) / self.generator_config.batch_size)
    
    
    def __getitem__(self, batch_index, batch_size=None):
        if batch_size == None:
            stu_indexes = self.stu_ids[(batch_index * self.generator_config.batch_size):((batch_index + 1) * self.generator_config.batch_size)]
        else:
            stu_indexes = self.stu_ids[(batch_index * batch_size):((batch_index + 1) * batch_size)]
        X0 = []
        X1 = []
        X2 = []
        X3 = []
        X4 = []
        X5 = []
        P = []
        Y = []
        
        for stu_index in stu_indexes:
            x0, x1, x2, x3, x4, x5, y, p = self.__data_generation(stu_index)
            X0.append(x0[np.newaxis, :, :])
            X1.append(x1[np.newaxis, :, :])
            X2.append(x2[np.newaxis, :, :])
            X3.append(x3[np.newaxis, :, :])
            X4.append(x4[np.newaxis, :, :])
            X5.append(x5[np.newaxis, :, :])
            Y.append(y[np.newaxis, :, :])
            P.append(p[np.newaxis, :])

        X0 = np.concatenate(X0, axis=0)
        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        X3 = np.concatenate(X3, axis=0)
        X4 = np.concatenate(X4, axis=0)
        X5 = np.concatenate(X5, axis=0)
        Y = np.concatenate(Y, axis=0)
        P = np.concatenate(P, axis=0)
        return [X0, X1, X2, X3, X4, X5, Y, P], []
    
    
    def on_epoch_end(self):
        if self.generator_config.shuffle == True:
            self.num_epoch += 1
            # Shuffle the order of inputs.
            np.random.seed(self.num_epoch)
            np.random.shuffle(self.stu_ids)
    
    
    def __data_generation(self, index):
        # x0: Course List.
        # x1: Multi-hot Major Vector.
        # x2: One-hot Relative Semester Vector.
        # x3: One-hot Semesters' Name Vector.
        # p, The place is masked or not.
        
        temp_seed = index if self.fixed_seed else (index + self.num_epoch)
        
        temp_start = self.data[index]['start']
        temp_courses = self.data[index]['courses'].astype(int)
        temp_majors = self.data[index]['majors'].astype(int)
        
        course_mat = list2mat(temp_courses, [self.num_semesters, self.num_courses])
        major_mat = list2mat(temp_majors, [self.num_semesters, self.num_majors])
        
        
        temp_retain_rate = np.random.rand(self.num_semesters)
        temp_mask = temp_retain_rate < self.mask_rate
        
        x0 = np.zeros(course_mat.shape)
        x0[temp_mask.astype(int)] = course_mat[temp_mask.astype(int)]
        y = np.zeros(course_mat.shape)
        y[(1 - temp_mask).astype(int)] = course_mat[(1 - temp_mask).astype(int)]
        x1 = np.tile(major_mat[0][np.newaxis, :], [self.num_semesters, 1])
        x2 = np.eye(self.num_semesters)
        x3 = np.zeros([self.num_semesters, 3])
        x3[np.arange(self.num_semesters), np.mod(np.arange(self.num_semesters) + temp_start, 3)] = 1
        x4 = onehot_encoding(x0.sum(1), [self.num_semesters, 15])
        p = temp_mask
        
        return x0, x1, x2, x3, x4, p, y