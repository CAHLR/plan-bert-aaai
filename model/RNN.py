from keras import regularizers
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Multiply, Lambda, Dense, Masking
from keras.optimizers import Adam
import numpy as np

from model.transformer_util.extras import TiedOutputEmbedding
from model.transformer_util.transformer import TransformerBlock
from model.transformer_util.multihot_utils import ReusableEmbed_Multihot
from model.transformer_util.attention import MultiHeadSelfAttention

from model.LossFunction import batch_crossentropy, confidence_penalty, recall_at_10


def LSTM(config):
    print(config)
    
    mask_future = config['mask_future'] # False
    vanilla_wiring = True # True
    use_parameter_sharing = False # False
    
    num_sem = config['num_sem']
    num_times = config['num_times']
    num_input_list = [config['num_input_0'],config['num_input_1'],config['num_input_2'],config['num_input_3']]
    num_input_list += config['num_stu_feat_list']
    num_input_list += config['num_crs_feat_list']
    
    num_layers = config['num_layers']
    embedding_dim = config['embedding_dim']
    num_heads = config['num_heads']
    
    transformer_dropout = config['transformer_dropout']
    embedding_dropout = config['embedding_dropout']
    l2_reg_penalty_weight = config['l2_reg_penalty_weight']
    confidence_penalty_weight = config['confidence_penalty_weight']
    lrate = config['lrate']
    
    # [WhetherTheFeatureIsUsed, DimOfFeature, Name, InputLayer]
    for iter, num_input in enumerate(num_input_list):
        num_input_list[iter].append(Input(shape=(num_sem+num_times,num_input[1]), dtype='float', name=num_input[2]))
    l2_reg = (regularizers.l2(l2_reg_penalty_weight) if l2_reg_penalty_weight else None)
    
    embedding, embedding_matrix = ReusableEmbed_Multihot(
            num_input_list[0][1], embedding_dim, input_length=num_sem+num_times,name=num_input_list[0][2]+'Embedding',embeddings_regularizer=l2_reg)(num_input_list[0][-1])
    embedding_list = [embedding]
    for num_input in num_input_list[1:]:
        if num_input[0] == True: # if the feature is used
            embedding_list.append(
                ReusableEmbed_Multihot(
                    num_input[1], embedding_dim, input_length=num_sem+num_times, name=num_input[2]+'Embedding'
                )(num_input[3])[0]
            )
    target = Input(shape=(num_sem+num_times, num_input_list[0][1]), dtype='float', name='Target')
    use_pred = num_input_list[3][3]
    
    next_step_input = Add(name='Embedding_Add')(embedding_list)
    if use_two_direction:
        for i in range(num_layers):
            next_step_input = Bidirectional(LSTM_layer(
                embedding_dim, dropout=lstm_dropout, return_sequences=True, name='Bi-LSTM_layer_{}'.format(i)
            ), merge_mode='sum')(next_step_input)
    else:
        for i in range(num_layers):
            next_step_input = LSTM_layer(
                embedding_dim, dropout=lstm_dropout, return_sequences=True, name='LSTM_layer_{}'.format(i))(next_step_input)

    predict = Softmax(name='prediction')(
        TiedOutputEmbedding(
            projection_regularizer=l2_regularizer,
            projection_dropout=embedding_dropout,
            name='prediction_logits')([next_step_input, embedding_matrix]))
    
    