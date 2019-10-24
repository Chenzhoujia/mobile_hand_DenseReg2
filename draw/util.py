'''
This file is modified on https://github.com/guohengkai/region-ensemble-network/blob/master/evaluation/util.py
'''
import os
import shutil

import numpy as np
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import numpy.linalg as alg

def get_positions_aw(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(positions), (-1, int(len(positions[0]) / 3), 3))

def get_positions(in_file):
    def string_handel(x):
        x = x[1:]
        return float(x)
    positions = []
    file_id = []
    with open(in_file) as f:
        for line in f:
            line = line.strip().split('\t')
            file_id.append(line[0])
            line = line[1:]
            line = map(float, line)
            positions.append(list(line))
    positions = np.array(positions)
    positions = np.reshape(positions, (len(file_id), -1,  3))
    return file_id, positions


def check_dataset(dataset):
    return dataset in set(['icvl', 'nyu', 'msra'])


def get_dataset_file(dataset):
    return 'groundtruth/{}/{}_test_groundtruth_label.txt'.format(dataset, dataset)


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.03, -587.07, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x

halfResX = 640/2
halfResY = 480/2
coeffX = 588.036865
coeffY = 587.075073

def world2pixel(x, fx = coeffX, fy = coeffY, ux = halfResX, uy = halfResY):

    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x


def get_errors(dataset, in_file):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    labels = get_positions(get_dataset_file(dataset))
    outputs = get_positions(in_file)
    params = get_param(dataset)
    labels = pixel2world(labels, *params)
    outputs = pixel2world(outputs, *params)
    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))
    return errors

def get_msra_viewpoint(in_file):
    with open(in_file) as f:
        viewpoint = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(viewpoint), (-1, 2))
def figure_joint_skeleton(dm, uvd_pt1,uvd_pt2,uvd_gt,base_path,id):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(dm, cmap=matplotlib.cm.Greys)
    ax1.axis('off')
    ax1.set_title("input image")
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(dm, cmap=matplotlib.cm.Greys)
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(dm, cmap=matplotlib.cm.Greys)
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(dm, cmap=matplotlib.cm.Greys)
    linewidth_ = 1
    for i in range(3):
        if i==0:
            ax = ax2
            fig_color = ['c', 'm', 'y', 'g', 'r']
            uvd_pt =uvd_gt
            ax.set_title("ground truth")
        elif i==1:
            ax = ax3
            fig_color = ['c', 'm', 'y', 'g', 'r']
            uvd_pt =uvd_pt1
            ax.set_title("with GNN")
        else:
            ax = ax4
            fig_color = ['c', 'm', 'y', 'g', 'r']
            uvd_pt = uvd_pt2
            ax.set_title("without GNN")
        ax.axis('off')
        for f in range(5):
            ax.plot([uvd_pt[f*2,0], uvd_pt[f*2+1,0]],
                    [uvd_pt[f*2,1], uvd_pt[f*2+1,1]], color=fig_color[f], linewidth=linewidth_)
            ax.scatter(uvd_pt[f*2,0],uvd_pt[f*2,1],s=3,c=fig_color[f])
            ax.scatter(uvd_pt[f*2+1,0],uvd_pt[f*2+1,1],s=3,c=fig_color[f])
            if f<4:
                ax.plot([uvd_pt[13,0], uvd_pt[f*2+1,0]],
                        [uvd_pt[13,1], uvd_pt[f*2+1,1]], color=fig_color[f], linewidth=linewidth_)
        ax.plot([uvd_pt[9,0], uvd_pt[10,0]],
                [uvd_pt[9,1], uvd_pt[10,1]], color='r', linewidth=linewidth_)

        ax.scatter(uvd_pt[13,0], uvd_pt[13,1], s=10, c='w')
        ax.scatter(uvd_pt[11,0], uvd_pt[11,1], s=5, c='b')
        ax.scatter(uvd_pt[12,0], uvd_pt[12,1], s=5, c='b')

        ax.plot([uvd_pt[13,0], uvd_pt[11,0]],
                [uvd_pt[13,1], uvd_pt[11,1]], color='b', linewidth=linewidth_)
        ax.plot([uvd_pt[13,0], uvd_pt[12,0]],
                [uvd_pt[13,1], uvd_pt[12,1]], color='b', linewidth=linewidth_)
        ax.plot([uvd_pt[13,0], uvd_pt[10,0]],
                [uvd_pt[13,1], uvd_pt[10,1]], color='r', linewidth=linewidth_)

    plt.savefig(base_path+'\\'+str(id).zfill(5)+'.png')
    return fig
def meanJntError(cls_obj, skel1, skel2):
    diff = skel1.reshape(-1,3) - skel2.reshape(-1,3)
    diff = alg.norm(diff, axis=1)
    mean = np.nanmean(diff)
    # if np.isnan(mean):
    #     print("get nan return 7.0")
    #     return 7.0
    return mean
def getbone_length(pose):
    # 计算每个骨骼长度 8252 14 3 -> 8252 13
    bone_len_res = np.zeros((8252, 13))
    for i in range(8252):
        bone_len_res[i, :] = getbone_length_(pose[i,:,:])
    return bone_len_res
def getbone_length_(pose):
    bone_len_res = np.zeros((13))
    # 计算每个骨骼长度 14 3 -> 13
    start = [0,2,4,6,8,9 ,1, 3, 5, 7, 10,13,13]
    end =   [1,3,5,7,9,10,13,13,13,13,13,11,12]
    for i in range(13):
        bone_len_res[i] = np.linalg.norm(pose[start[i]] - pose[end[i]])
    return bone_len_res
if __name__ == '__main__':
    file, result = get_positions("F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\result222\\27403\\testing-2019-10-12_12_29_34.872939-result.txt")
    result = world2pixel(result)

    groundtruth = get_positions_aw("F:\\chen\\pycharm\\awesome-hand-pose-estimation-master\\evaluation\\groundtruth\\nyu\\nyu_test_groundtruth_label.txt")
    result2 = get_positions_aw("F:\\chen\\pycharm\\awesome-hand-pose-estimation-master\\evaluation\\results\\nyu\\CVPR18_NYU_denseReg.txt")

    # 计算每个骨骼长度 8252 14 3 -> 8252 13
    groundtruth_bone = getbone_length(groundtruth)
    # 按列统计方差
    print(np.std(groundtruth_bone, axis=0))

    groundtruth_bone = getbone_length(result)
    # 按列统计方差
    print(np.nanstd(groundtruth_bone, axis=0))

    groundtruth_bone = getbone_length(result2)
    # 按列统计方差
    print(np.nanstd(groundtruth_bone, axis=0))

    # # 根据误差 从大到小排序
    # errors = groundtruth[:,:,:2] - result[:,:,:2]
    # errors = errors.reshape(-1,2)
    # errors = np.linalg.norm(errors,axis=1,keepdims=True)
    # errors = errors.reshape((8252,14))
    # errors = np.nanmean(errors,1)
    # y = errors.argsort()
    # y = y[::-1]
    # id = 0
    #
    # base_path = 'F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\image\\result'
    # if os.path.exists(base_path):
    #     shutil.rmtree(base_path)
    # os.makedirs(base_path)
    # for i in range(80):
    #     one_file = y[i*100+2]
    #     file_name = file[one_file]
    #     file_name = "F:\\chen\\pycharm\\dataset\\dataset\\test\\"+file_name[2:-1]
    #     print(file_name)
    #     lena = mpimg.imread(file_name) # 读取和代码处于同一目录下的 lena.png
    #     # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    #     figure_joint_skeleton(lena, result[one_file],result2[one_file], groundtruth[one_file],base_path,id)
    #     id = id+1
