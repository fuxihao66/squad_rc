import argparse
import json
import math
import os
import shutil
from pprint import pprint
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from model import *
from train import *
from metadata_operation import *
from pre_processing import *
from rouge_operation import *
def main(config):

    
    if config.mode == 'train':
        _train(config)
    # elif config.mode == 'test':
    #     _test(config)
    # elif config.mode == 'forward':
    #     _forward(config)
    else:
        raise ValueError("invalid value for 'mode': {}".format(config.mode))



def _train(config):
	path = '''/home/zhangs/RC/SQUAD_data/train-v1.1.json'''
	train_data_dict = read_metadata(path)
	train_data_dict = renew_data_dict(train_data_dict)
	

	'''TODO: the char dict should also contain dev-set'''
	char2idx_dict, char_vocabulary_size = get_char2idx(train_data_dict)
    
	path_vocab = '''/home/zhangs/RC/squad_rc/vocabulary.txt'''
	word2idx_dict, emb_mat = get_word2idx_and_embmat('''/home/zhangs/RC/data/glove.6B.100d.txt''', path_vocab)
    
	train_data = DataSet(train_data_dict)
	train_data.init_without_ans(config.batch_size, 'train')
    
	config.emb_mat = emb_mat
	config.char_vocab_size = char_vocabulary_size

	with tf.name_scope("model"):
		model = Model(config, word2idx_dict, char2idx_dict)
    # models = get_multi_models(config, word2idx_dict, char2idx_dict)

	with tf.name_scope("trainer"):
		trainer = single_GPU_trainer(config, model)
    # trainer = MultiGPUTrainer(config, models)

    
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    

	global_step = 0


	train_writer = tf.summary.FileWriter('/home/zhangs/RC/SQUAD_data/FINAL', sess.graph)

	init = tf.global_variables_initializer()
	sess.run(init)

	batch_list = train_data.get_batch_list()
	batch_list_length = len(batch_list)
	batch_num = 5


	for i in range(config.num_epochs):

		for i in range(int(math.ceil(batch_list_length/batch_num))):
        # for i in range(int(math.ceil(batch_list_length/config.num_gpus))):
			sub_batch_list = get_random_eles_from_list(batch_list, batch_num)
            # sub_batch_list = get_random_eles_from_list(batch_list, config.num_gpus)

            # global_step = sess.run(models[0].global_step) + 1
            # print(global_step)
            # if global_step == 10000:
            #     trainer.change_lr(0.3)
            # loss, summary, train_op = trainer.step(sess, sub_batch_list, True)
            # train_writer.add_summary(summary, global_step)
            # print(loss)

			for batch in sub_batch_list:
				global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
				get_summary = True
				print(global_step)

				if global_step == 10000:
					trainer.change_lr(0.1)
				if global_step == 20000:
					trainer.change_lr(0.04)

				loss, summary, train_op = trainer.step(sess, batch, get_summary=get_summary)
				train_writer.add_summary(summary, global_step)

				print(loss)


	'''start to evaluate via dev-set'''
	dev_data_dict = read_metadata('''/home/zhangs/RC/SQUAD_data/dev-v1.1.json''')
	dev_data_dict_backup = read_metadata('''/home/zhangs/RC/SQUAD_data/dev-v1.1.json''')
	dev_data   = DataSet(dev_data_dict)
	dev_data.init_without_ans(config.batch_size, 'dev')
	# ans_list = dev_data.answers_list
	dev_batches = dev_data.get_batch_list()

	summaries = []
	for j, batch in enumerate(dev_batches):
		feed_dict = model.get_feed_dict(batch, None, None, False)
		yp, yp2 = sess.run([model.yp, model.yp2], feed_dict=feed_dict)   
		yp = get_y_index(yp)
		yp2= get_y_index(yp2)
		for i in range(len(batch['x'])):
                        
			words = batch['x'][i]

			try:
				summary = get_phrase(dev_data_dict_backup['passages'][j*config.batch_size+i], words, [yp[i], yp2[i]])
				summaries.append(summary)  
			except:
				print(yp[i])
				print(yp2[i])
				print(dev_data_dict_backup['passages'][j*config.batch_size+i])
				print(words)
    
	print(len(summaries))
	
	path_result = '''/home/zhangs/RC/SQUAD_data/out_second_time/dev_out.txt'''
	with open(path_result, 'w') as out_file:
		for summary in summaries:
			if '\n' in summary:
				summary = summary[:summary.index('\n')]
			out_file.write(summary+'\n')

    
            
