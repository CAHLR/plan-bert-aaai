from model.multihot_utils import recall_at_10
import keras.callbacks as callbacks
import math, copy
import util.Metrics as Metrics
import numpy as np


class CosineLRSchedule:
    """
    Cosine annealing with warm restarts, described in paper
    "SGDR: stochastic gradient descent with warm restarts"

    Changes the learning rate, oscillating it between `lr_high` and `lr_low`.
    It takes `period` epochs for the learning rate to drop to its very minimum,
    after which it quickly returns back to `lr_high` (resets) and everything
    starts over again.

    With every reset:
        * the period grows, multiplied by factor `period_mult`
        * the maximum learning rate drops proportionally to `high_lr_mult`

    This class is supposed to be used with
    `keras.callbacks.LearningRateScheduler`.
    """
    def __init__(self, lr_high: float, lr_low: float, initial_period: int = 50,
                 period_mult: float = 2, high_lr_mult: float = 0.97):
        self._lr_high = lr_high
        self._lr_low = lr_low
        self._initial_period = initial_period
        self._period_mult = period_mult
        self._high_lr_mult = high_lr_mult

    def __call__(self, epoch, lr):
        return self.get_lr_for_epoch(epoch)

    def get_lr_for_epoch(self, epoch):
        assert epoch >= 0
        t_cur = 0
        lr_max = self._lr_high
        period = self._initial_period
        result = lr_max
        for i in range(epoch + 1):
            if i == epoch:  # last iteration
                result = (self._lr_low +
                          0.5 * (lr_max - self._lr_low) *
                          (1 + math.cos(math.pi * t_cur / period)))
            else:
                if t_cur == period:
                    period *= self._period_mult
                    lr_max *= self._high_lr_mult
                    t_cur = 0
                else:
                    t_cur += 1
        return result


def fit(model, train_generator, valid_generator, epoch_limit=200, loss_nonimprove_limit=3, batch_size=32, use_cosine_lr=True, model_save_path=None):
    # Train model with early stopping condition
    
    metric = 'recall_at_10'
    print('Training model...')
    base_logger = callbacks.BaseLogger(stateful_metrics=['recall_at_10', 'val_recall_at_10'])
    early_stopping = callbacks.EarlyStopping(monitor='val_recall_at_10', patience=loss_nonimprove_limit, verbose=1, mode='max')
    model_callbacks = [base_logger, early_stopping]

    if use_cosine_lr:
        model_callbacks.append(callbacks.LearningRateScheduler(
        CosineLRSchedule(lr_high=1e-4, lr_low=1e-4 / 32, initial_period=10), verbose=1))
    if model_save_path is not None:
        model_callbacks.append(callbacks.ModelCheckpoint(model_save_path, monitor='val_recall_at_10', mode='max', save_best_only=True, verbose=True))

    model_history = model.fit_generator(
        generator=train_generator, 
        validation_data=valid_generator,  
        epochs=epoch_limit, 
        callbacks=model_callbacks,
        use_multiprocessing=True,
        workers=10)

    best_accuracy = max(model_history.history[metric])
    print("Best accuracy:", best_accuracy)
    

def test(model, generator, filter, duplicate_filter='pass', history_num=0):
    target_all = []
    predict_all = []
    for iter, batch in enumerate(generator):
        if iter == len(generator): break
        target = batch[0][-1]
        year_filter, union_filter = batch[1]
        predict = model.predict_on_batch(batch[0])[0]
        target_all.append(target[:, history_num:generator.num_sem])
        if filter == 'year':
            predict = predict[:, history_num:generator.num_sem] * year_filter[:, history_num:generator.num_sem]
        elif filter == 'union':
            predict = predict[:, history_num:generator.num_sem] * union_filter[:, history_num:generator.num_sem]
        else:
            predict = predict[:, history_num:generator.num_sem]
            
        if duplicate_filter == 'v1':
            stu_id, semester_id, course_id = np.where(batch[0][0] >= 1)
            predict[stu_id, :, course_id] = 0
            for sem in range(predict.shape[1]):
                temp_rank = (-predict[:, sem]).argsort(axis=-1) # [num_stu, course_ids]
                for stu in range(predict.shape[0]):
                    predict[stu, sem+1:, temp_rank[:4]] = 0
        elif duplicate_filter == 'v2':
            stu_id, semester_id, course_id = np.where(batch[0][0] >= 1)
            predict[stu_id, :, course_id] = 0
            #temp_rank = (-predict).argsort(axis=1) # [num_stu, sem, course_ids]
        else:
            pass
        
        predict_all.append(predict)
    target_all = np.concatenate(target_all, axis=0)
    predict_all = np.concatenate(predict_all, axis=0)
    
    recall_mean, recall_per_sem = Metrics.recall(target_all, predict_all, at_n=10)
    #print('recall@10 per sem : ')
    #print(np.round(recall_per_sem, 4))
    #print('recall@10 average : ')
    #print(np.round(recall_mean, 4))
    
    #recall_mean, recall_per_sem = Metrics.recall(target_all, predict_all, at_n=0)
    #print('recall@N per sem : ')
    #print(np.round(recall_per_sem, 4))
    #print('recall@N average : {}'.format(np.round(recall_mean, 4)))
    #print(np.round(recall_mean, 4))
    
    return target_all, predict_all, recall_mean, recall_per_sem
'''

def test(model, generator, filter, auto_regressive=False, history_num=0, use_schedule=False,
    top_k_select=True, prereq_filter=False, taken_filter=True, idx2course=None, db=None, filter_output=False,
    course_dept2idx={}, course_subj2idx={}, cid2subj={}, cid2dept={}):
    target_all = []
    predict_all = []
    if not auto_regressive:
        for iter, batch in enumerate(generator):
            if iter == len(generator): break
            target = batch[0][-1]
            year_filter, union_filter = batch[1]
            predict = model.predict_on_batch(batch[0])[0]
            target_all.append(target[:, history_num:])
            if filter == 'year':
                predict_all.append(predict[:, history_num:] * year_filter[:, history_num:])
            elif filter == 'union':
                predict_all.append(predict[:, history_num:] * union_filter[:, history_num:])
            else:
                predict_all.append(predict[:, history_num:])
    else:
        for iter, batch in enumerate(generator):
            if iter == len(generator): break
            target = batch[0][-1]
            S = batch[1][0]
            outputs = []
            for sem in range(history_num + 1, 12):
                predict = model.predict_on_batch(batch[0])[0]

                next_input = predict[:, sem - 1]
                if top_k_select:
                    indices = np.argsort(-next_input, axis=-1)
                    # select top 4 courses for use
                    if taken_filter:
                        sel_indices = []
                        o_indices = []
                        for i in range(indices.shape[0]):
                            sel = []
                            o_is = []
                            taken = set(np.where(batch[0][0][i, :, :])[1].tolist())
                            idx = 0
                            while len(o_is) < 10 and idx < indices.shape[1]:
                                cond = not prereq_filter
                                if prereq_filter:
                                    cond = filter_prereq(indices[i, idx], list(taken), idx2course, db)
                                if indices[i, idx] not in taken and cond:
                                    if len(sel) < 4:
                                        sel.append(indices[i, idx])
                                    o_is.append(indices[i, idx])
                                idx += 1
                            sel_indices.append(sel)
                            o_indices.append(o_is)
                        sel_indices = np.array(sel_indices)
                        o_indices = np.array(o_indices)
                    else:
                        sel_indices = indices[:, 0:4]
                        o_indices = indices[:, 0:10]
                    new_next_input = np.zeros(shape=next_input.shape)
                    new_o = np.zeros(shape=next_input.shape)
                    
                    for i in range(sel_indices.shape[0]):
                        new_next_input[i, sel_indices[i]] = 1.
                        new_o[i, o_indices[i]] = 0.1
                    #new_next_input[:, sel_indices] = 1.
                    next_input = new_next_input
                    
                    if filter_output:
                        outputs.append(new_o)
                        # also impute courses department and subject for autoregressive input
                        for i in range(indices.shape[0]):
                            cids = [idx2course[str(id)] for id in sel_indices[i]]
                            depts = [cid2dept[cid] for cid in cids if cid in cid2dept]
                            subjs = [cid2subj[cid] for cid in cids if cid in cid2subj]
                            dept_idxes = np.array([course_dept2idx[dept] for dept in depts])
                            subj_idxes = np.array([course_subj2idx[subj] for subj in subjs])
                            batch[0][3][i, sem, dept_idxes] = 1.
                            batch[0][4][i, sem, subj_idxes] = 1.
                batch[0][0][:, sem] = next_input
            if filter_output:
                predict = np.zeros(shape=predict.shape)
                predict[:, history_num:len(outputs) + history_num] = np.array(outputs).transpose(1, 0, 2)
            target_all.append(target[:, history_num:])
            if use_schedule:
                predict_all.append(predict[:, history_num:] * S[:, history_num:])
            else:
                predict_all.append(predict[:, history_num:])
                
    target_all = np.concatenate(target_all, axis=0)
    predict_all = np.concatenate(predict_all, axis=0)
    recall_mean, recall_per_sem = Metrics.recall(target_all, predict_all, at_n=10)
    print('recall@10 per sem : ')
    print(np.round(recall_per_sem, 4))
    print('recall@10 average : ')
    print(np.round(recall_mean, 4))
    
    #recall_mean, recall_per_sem = Metrics.recall(target_all, predict_all, at_n=0)
    #print('recall@N per sem : ')
    #print(np.round(recall_per_sem, 4))
    #print('recall@N average : {}'.format(np.round(recall_mean, 4)))
    #print(np.round(recall_mean, 4))
    
    return target_all, predict_all, recall_mean, recall_per_sem
'''
def batch(generator):
    for batch in generator:
        return batch
        
        
import pynvml   
def cuda_max_free(cuda_list):
    pynvml.nvmlInit()
    free_list = []
    for iter in cuda_list:
        handle = pynvml.nvmlDeviceGetHandleByIndex(iter)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_list.append(meminfo.free)
    return cuda_list[np.array(free_list).argmax()]