import numpy as np
import copy
import tqdm

def recall(target, predict, at_n):
    num_students = target.shape[0]
    num_semesters = target.shape[1]
    num_courses = target.shape[2]
    
    #predict += np.random.rand(*predict.shape)+1e-10
    
    hit_mat = np.zeros(predict.shape)
    for sample in range(num_students):
        for semester in range(num_semesters):
            n_temp = target[sample, semester].sum()
            if len(np.shape(at_n)) == 1: # recall@n on average
                index = np.argsort(-predict[sample, semester])[:int(at_n[semester])]
            elif at_n == 0: # recall@n
                index = np.argsort(-predict[sample, semester])[:int(n_temp)]
            elif at_n > 0: # recall@10
                index = np.argsort(-predict[sample, semester])[:int(at_n)]
            hit_mat[sample, semester, index] = 1
    hit_mat = (hit_mat*target)
    
    recall_mean = np.round(hit_mat).sum() / np.round(target).sum()
    recall_per_sem = hit_mat.sum(0).sum(-1) / target.sum(0).sum(-1)
    
    return np.round(recall_mean, 4), np.round(recall_per_sem, 4)

def recall_year(target, predict, at_n):
    num_students = target.shape[0]
    num_semesters = target.shape[1]
    num_courses = target.shape[2]
    
    hit_mat = np.zeros(predict.shape)
    for sample in range(num_students):
        for semester in range(num_semesters):
            n_temp = target[sample, semester].sum()
            if len(np.shape(at_n)) == 1: # recall@n on average
                thresh = np.sort(predict[sample, semester])[-int(at_n[semester])]
                hit_mat[sample, semester] = (predict[sample, semester] >= thresh).astype(float)
            elif at_n == 0: # recall@n
                thresh = np.sort(predict[sample, semester])[-int(n_temp)]
                hit_mat[sample, semester] = (predict[sample, semester] >= thresh).astype(float)
            elif at_n > 0: # recall@10
                thresh = np.sort(predict[sample, semester])[-int(at_n)]
                hit_mat[sample, semester] = (predict[sample, semester] >= thresh).astype(float)
    
    hit_mat = (hit_mat*target)
    
    recall_mean = np.round(hit_mat).sum() / np.round(target).sum()
    recall_per_year = []
    for iter in range(4):
        recall_per_year.append(np.round(hit_mat[:, (iter*3):(iter*3+3)]).sum() / np.round(target[:, (iter*3):(iter*3+3)]).sum())
    
    return np.round(recall_mean, 4), np.round(recall_per_year, 4)


def flat_recall(target, predict, at_n):
    assert np.all(target.shape == predict.shape)
    assert len(np.shape(at_n)) == 0
    
    num_students = target.shape[0]
    num_semesters = target.shape[1]
    num_courses = target.shape[2]
    
    batch_size = 32
    recall_list = []
    
    for iter in tqdm.tqdm(range(0, num_students, batch_size), ascii=True, ncols=60):
        predict_2 = copy.deepcopy(predict[iter:iter+batch_size])
        target_2 = copy.deepcopy(target[iter:iter+batch_size])
        for sample in range(batch_size):
            predict_3 = predict_2[sample].reshape(-1)
            target_3 = target_2[sample].reshape(-1)
            
            if np.sum(target_3) > 0:
                predict_id = np.zeros(predict_3.shape)
                predict_id[np.argsort(predict_3)[int(-at_n):]] = 1
            
                temp_recall = np.sum(predict_id * target_3) / np.sum(target_3)
            
                recall_list.append(temp_recall)
    return np.round(np.mean(recall_list), 4)