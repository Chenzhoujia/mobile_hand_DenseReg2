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
from tqdm import tqdm

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

halfResX = 640/2
halfResY = 480/2
coeffX = 588.036865
coeffY = 587.075073

def pixel2world(x,fx = coeffX, fy = coeffY, ux = halfResX, uy = halfResY):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x

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
def figure_joint_skeleton(dm2, uvd_pt1,uvd_pt2,uvd_gt,base_path,id):

    # 根据 uvd_pt1 uvd_pt2 uvd_gt 中的 (0,1) 维度计算 中间坐标 以及尺度
    x_mid = np.concatenate([uvd_gt[:,0]], axis=0)
    y_mid = np.concatenate([uvd_gt[:,1]], axis=0)

    x_max = np.nanmax(x_mid)
    x_min = np.nanmin(x_mid)
    y_max = np.nanmax(y_mid)
    y_min = np.nanmin(y_mid)

    x_mid = x_max/2.0+x_min/2.0
    y_mid = y_max/2.0+y_min/2.0
    scale = np.max([x_max-x_min, y_max-y_min])
    scale = int(scale*0.7)

    # 对dm 进行 切割
    if int(y_mid)-scale<0 or int(x_mid)-scale<0:
        return

    dm = dm2[int(y_mid)-scale:int(y_mid)+scale, int(x_mid)-scale:int(x_mid)+scale, :]

    r = dm[:,:,0]
    g = dm[:,:,1]
    b = dm[:,:,2]
    d = g*256+b
    d = d/np.max(d)
    dm[:,:,0] = d
    dm[:,:,1] = d
    dm[:,:,2] = d


    fig = plt.figure(figsize=(6, 24))
    ax1 = fig.add_subplot(4,1,1)
    ax1.imshow(dm, cmap=matplotlib.cm.Greys)
    ax1.axis('off')
    # ax1.set_title("input image")
    ax2 = fig.add_subplot(4,1,2)
    ax2.imshow(dm, cmap=matplotlib.cm.Greys)
    ax3 = fig.add_subplot(4,1,3)
    ax3.imshow(dm, cmap=matplotlib.cm.Greys)
    ax4 = fig.add_subplot(4,1,4)
    ax4.imshow(dm, cmap=matplotlib.cm.Greys)
    linewidth_ = 0.3
    for i in range(3):
        if i==0:
            ax = ax2
            fig_color = ['c', 'm', 'y', 'g', 'r']
            uvd_pt =uvd_gt
            # ax.set_title("ground truth")
        elif i==1:
            ax = ax3
            fig_color = ['c', 'm', 'y', 'g', 'r']
            uvd_pt =uvd_pt1
            # ax.set_title("with GNN")
        else:
            ax = ax4
            fig_color = ['c', 'm', 'y', 'g', 'r']
            uvd_pt = uvd_pt2
            # ax.set_title("without GNN")
        ax.axis('off')

        uvd_pt[:,0]  = uvd_pt[:,0] - x_mid + scale
        uvd_pt[:,1]  = uvd_pt[:,1] - y_mid + scale
        size = 10
        for f in range(5):
            ax.plot([uvd_pt[f*2,0], uvd_pt[f*2+1,0]],
                    [uvd_pt[f*2,1], uvd_pt[f*2+1,1]], color=fig_color[f], linewidth=linewidth_*size)
            ax.scatter(uvd_pt[f*2,0],uvd_pt[f*2,1],s=6*size,c=fig_color[f])
            ax.scatter(uvd_pt[f*2+1,0],uvd_pt[f*2+1,1],s=6*size,c=fig_color[f])
            if f<4:
                ax.plot([uvd_pt[13,0], uvd_pt[f*2+1,0]],
                        [uvd_pt[13,1], uvd_pt[f*2+1,1]], color=fig_color[f], linewidth=linewidth_*size)
        ax.plot([uvd_pt[9,0], uvd_pt[10,0]],
                [uvd_pt[9,1], uvd_pt[10,1]], color='r', linewidth=linewidth_*size)

        ax.scatter(uvd_pt[13,0], uvd_pt[13,1], s=20*size, c='b')
        ax.scatter(uvd_pt[11,0], uvd_pt[11,1], s=10*size, c='b')
        ax.scatter(uvd_pt[12,0], uvd_pt[12,1], s=10*size, c='b')

        ax.plot([uvd_pt[13,0], uvd_pt[11,0]],
                [uvd_pt[13,1], uvd_pt[11,1]], color='b', linewidth=linewidth_*size)
        ax.plot([uvd_pt[13,0], uvd_pt[12,0]],
                [uvd_pt[13,1], uvd_pt[12,1]], color='b', linewidth=linewidth_*size)
        ax.plot([uvd_pt[13,0], uvd_pt[10,0]],
                [uvd_pt[13,1], uvd_pt[10,1]], color='r', linewidth=linewidth_*size)

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
def compute_pose_error(pose):
    error1 = pose
    error1 = error1.reshape((-1,3))
    error1 = np.linalg.norm(error1, axis=1, keepdims=True)
    error1 = error1.reshape((8250,14))
    error1 = np.nanmean(error1)
    return error1

def add_method_name(method_name, path_name):
    # method_name.append('DeepPrior')
    # path_name.append('\\results\\nyu\\CVWW15_NYU_Prior.txt')
    #
    # method_name.append('DeepPrior-Refine')
    # path_name.append('\\results/nyu/CVWW15_NYU_Prior-Refinement.txt')
    #
    # method_name.append('Feedback')
    # path_name.append('\\results\\nyu\\ICCV15_NYU_Feedback.txt')
    #
    # method_name.append('DeepModel')
    # path_name.append('\\results\\nyu\\IJCAI16_NYU_DeepModel.txt')
    #
    # method_name.append('Lie-X')
    # path_name.append('\\results\\nyu\\IJCV16_NYU_LieX.txt')
    #
    # method_name.append('3DCNN')
    # path_name.append('\\results\\nyu\\CVPR17_NYU_3DCNN.txt')
    #
    # method_name.append('Guo_Baseline')
    # path_name.append('\\results\\nyu\\ICIP17_NYU_Guo_Basic.txt')
    #
    # method_name.append('REN-4x6x6')
    # path_name.append('\\results\\nyu\\ICIP17_NYU_REN_4x6x6.txt')
    #
    # method_name.append('REN-9x6x6')
    # path_name.append('\\results\\nyu\\JVCI18_NYU_REN_9x6x6.txt')
    #
    # method_name.append('DeepPrior++')
    # path_name.append('\\results\\nyu\\ICCVW17_NYU_DeepPrior++.txt')
    #
    # method_name.append('Pose-REN')
    # path_name.append('\\results\\nyu\\NEUCOM18_NYU_Pose_REN.txt')
    #
    # method_name.append('DenseReg')
    # path_name.append('\\results\\nyu\\CVPR18_NYU_denseReg.txt')
    #
    # method_name.append('V2V-PoseNet')
    # path_name.append('\\results\\nyu\\CVPR18_NYU_V2V_PoseNet.txt')
    #
    # method_name.append('FeatureMapping')
    # path_name.append('\\results\\nyu\\CVPR18_NYU_DeepPrior++_FM.txt')
    #
    # method_name.append('SHPR-Net')
    # path_name.append('\\results\\nyu\\Access18_NYU_SHPR_Net_frontal.txt')
    #
    # method_name.append('SHPR-Net-three-views)')
    # path_name.append('\\results\\nyu\\Access18_NYU_SHPR_Net_three.txt')
    #
    # method_name.append('DeepHPS')
    # path_name.append('\\results\\nyu\\3DV18_NYU_DeepHPS.txt')
    #
    # method_name.append('HandPointNet')
    # path_name.append('\\results\\nyu\\CVPR18_NYU_HandPointNet.txt')
    #
    # method_name.append('Point-to-Point')
    # path_name.append('\\results\\nyu\\ECCV18_NYU_Point-to-Point.txt')
    #
    # method_name.append('MURAUER')
    # path_name.append('\\results\\nyu\\WACV19_NYU_murauer_n72757_uvd.txt')
    #
    # method_name.append('Generalized-Feedback')
    # path_name.append('\\results\\nyu\\TPAMI19_NYU_Generalized_Feedback.txt')
    #
    method_name.append('CrossInfoNet')
    path_name.append('\\results\\nyu\\CVPR19_NYU_CrossInfoNet.txt')

if __name__ == '__main__':
    file1, result1 = get_positions("F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\result222\\testing-2019-10-24_10_46_00.507492-result.txt")
    result1 = world2pixel(result1)
    groundtruth = get_positions_aw("F:\\chen\\pycharm\\awesome-hand-pose-estimation-master\\evaluation\\groundtruth\\nyu\\nyu_test_groundtruth_label.txt")
    groundtruth = groundtruth[:8250]



    # 根据误差 从大到小排序
    errors1 = groundtruth[:,:,:2] - result1[:,:,:2]
    errors1 = errors1.reshape(-1,2)
    errors1 = np.linalg.norm(errors1,axis=1,keepdims=True)
    errors1 = errors1.reshape((8250,14))
    errors1 = np.nanmean(errors1,1)
    y1 = errors1.argsort()
    y1 = y1[::-1]

    # 排队获取
    method_name = []
    path_name = []
    add_method_name(method_name, path_name)

    for idx in tqdm(range(len(path_name))):

        result2 = get_positions_aw("F:\\chen\\pycharm\\awesome-hand-pose-estimation-master\\evaluation" + path_name[idx])
        result2 = result2[:8250]
        errors2 = groundtruth[:,:,:2] - result2[:,:,:2]
        errors2 = errors2.reshape(-1,2)
        errors2 = np.linalg.norm(errors2,axis=1,keepdims=True)
        errors2 = errors2.reshape((8250,14))
        errors2 = np.nanmean(errors2,1)
        y2 = errors2.argsort()
        y2 = y2[::-1]


        id = 0

        base_path = 'F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\image\\result\\' + method_name[idx]
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        os.makedirs(base_path)
        for i in range(5):
            one_file = y2[i] # 获取ID
            file_name = file1[one_file]
            file_name = "F:\\chen\\pycharm\\dataset\\dataset\\test\\"+file_name[10:-1]
            print(file_name)
            lena = mpimg.imread(file_name) # 读取和代码处于同一目录下的 lena.png
            # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
            figure_joint_skeleton(lena, result1[one_file],result2[one_file], groundtruth[one_file],base_path,id)
            id = id+1

        for i in range(49,-1,-1):
            one_file = y1[i] # 获取ID
            file_name = file1[one_file]
            file_name = "F:\\chen\\pycharm\\dataset\\dataset\\test\\"+file_name[10:-1]
            print("*" + file_name)
            lena = mpimg.imread(file_name) # 读取和代码处于同一目录下的 lena.png
            # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
            figure_joint_skeleton(lena, result1[one_file],result2[one_file], groundtruth[one_file],base_path,id)
            id = id+1

