from __future__ import print_function, absolute_import, division

#import gpu_config
import tensorflow as tf
import network.slim as slim
import numpy as np
import time, os
import cv2
from datetime import datetime
from data.evaluation import Evaluation

FLAGS = tf.app.flags.FLAGS

def test(model, selected_step):
    with tf.Graph().as_default():
        total_test_num = model.val_dataset.exact_num

        dms, poses, cfgs, coms, names = model.batch_input_test(model.val_dataset)
        model.test(dms, poses, cfgs, coms, reuse_variables=None)

        # dms, poses, names = model.batch_input_test(model.val_dataset)
        # model.test(dms, poses, reuse_variables=None)

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        if selected_step is not None:

            # # 得到该网络中，所有可以加载的参数
            retrain = False
            if retrain:
                variables = tf.contrib.framework.get_variables_to_restore()
                # 删除output层中的参数
                variables_to_resotre = [v for v in variables if 'GNN' not in v.name]
                # 构建这部分参数的saver
                saver = tf.train.Saver(variables_to_resotre)
                checkpoint_path = os.path.join(model.train_dir, 'model.ckpt-%d' % selected_step)
                saver.restore(sess, checkpoint_path)
                # 恢复saver
                saver = tf.train.Saver(tf.global_variables())
                print('[test_model]model has been resotored from %s' % checkpoint_path)
            else:
                checkpoint_path = os.path.join(model.train_dir, 'model.ckpt-%d'%selected_step)
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, checkpoint_path)
                print('[test_model]model has been resotored from %s'%checkpoint_path)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            model.summary_dir+'_'+model.val_dataset.subset,
            graph=sess.graph)
        
        res_path = os.path.join(model.train_dir, '%s-%s-result'%(model.val_dataset.subset, datetime.now()))
        res_path = res_path.replace(' ', '_')
        res_path = res_path.replace(':', '_')
        res_txt_path = res_path+'.txt'
        if os.path.exists(res_txt_path):
            os.remove(res_txt_path)
        err_path = res_path+'_error.txt'
        f = open(res_txt_path, 'w')

        # res_vid_path = res_path+'.avi'
        # codec = cv2.cv.CV_FOURCC('X','V','I','D')
        # the output size is defined by the visualization tool of matplotlib
        # vid = cv2.VideoWriter(res_vid_path, codec, 25, (640, 480))
        
        print('[test_model]begin test')
        test_num = 0
        step = 0
        maxJntError = []
        meanJntError = []
        while True:
            start_time = time.time()
            try:
                gt_vals, xyz_vals, valid_hms, name_vals = model.do_test(sess, summary_writer, step, names)
            except tf.errors.OutOfRangeError:
                print('run out of range')
                break
            if b'depth_1_0005744.png' in name_vals:
                # 保存变量
                valid_hms_all = np.concatenate((np.expand_dims(valid_hms[0], axis=0),
                                np.expand_dims(valid_hms[1], axis=0),
                                np.expand_dims(valid_hms[2], axis=0),
                                np.expand_dims(valid_hms[3], axis=0)), axis=0)
                # 存储
                np.save(file="F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\image\\hm\\data.npy", arr=valid_hms_all)

                # # 读取
                # b = np.load(file="data.npy")
            duration = time.time()-start_time
            
            for xyz_val, gt_val, name_val in zip(xyz_vals, gt_vals, name_vals):
                maxJntError.append(Evaluation.maxJntError(xyz_val, gt_val))
                meanJntError.append(Evaluation.meanJntError(xyz_val, gt_val))
                xyz_val = xyz_val.tolist()
                res_str = '%s\t%s\n'%(name_val, '\t'.join(format(pt, '.4f') for pt in xyz_val))
                res_str = res_str.replace('/', '\\')
                f.write(res_str)
                # vid.write(vis_val)
                test_num += 1
                if test_num >= total_test_num:
                    meanJntError_all = mean(meanJntError)
                    mean_error = str(mean(meanJntError))
                    num_test = str(len(meanJntError))
                    print("mean_error" + mean_error)
                    print("num_test" + num_test)
                    print('finish test')
                    f.close()
                    Evaluation.plotError(maxJntError, err_path)

                    return
            f.flush()
            
            if step%50 == 0:
                print('[%s]: %d/%d computed, with %.2fs'%(datetime.now(), step, model.max_steps, duration))
                mean_error = str(mean(meanJntError))
                num_test = str(len(meanJntError))
                print("mean_error"+mean_error)
                print("num_test"+num_test)

            step += 1
            if step == 275:
                break

        meanJntError_all = mean(meanJntError)
        print("meanJntError_min: " + str(meanJntError_all))
        print('finish test')
        f.close()
        Evaluation.plotError(maxJntError, 'result.txt')
def mean(a):
    return sum(a) / len(a)