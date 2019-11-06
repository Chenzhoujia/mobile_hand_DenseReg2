import shutil
import numpy as np
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from tqdm import tqdm
import os

def figure_heatmap(hm):
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(hm, cmap=matplotlib.cm.jet)
    fig.colorbar(im)
    return fig
def std_noise(im):
    im_ = im.reshape(-1)
    im_ = np.sort(im_)
    im_ = im_[::-1]
    im_[0:9] = 0
    return str(np.mean(im_))
def setDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

# filename = "2014"
#filename = "2054"
# filename = "3663"
filename = "5751"
base_file_dir="F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\image\\train_tmp\\"
base_file_dir_im="F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\image\\train_tmp_image\\"

# 给定编号
orig = 10000
big_id = 8
# 给定子编号
son_id = 2
# 给定 batch id
batch_id_list = [0,2,6,7]
for batch_id in batch_id_list:
    # 绘制： 绘制深度图、 绘制hm 14、 绘制hm相加起来14 、 CAM 、 自己加起来看看
    file_name = str(orig+big_id).zfill(7) + "_" + str(son_id)
    file_namer_im = str(orig+big_id).zfill(7) + "_" + str(son_id) + "_" + str(batch_id)
    setDir(base_file_dir_im + file_namer_im)
    dm_inputs_train_ = np.load(file=base_file_dir + file_name + "dm_inputs_train_.npy")
    CAM_hm_train_ = np.load(file=base_file_dir + file_name + "CAM_hm_train_.npy")
    hm_add_train_ = np.load(file=base_file_dir + file_name + "hm_add_train_.npy")
    valid_hm_train_ = np.load(file=base_file_dir + file_name + "valid_hm_train_.npy")

    dm_inputs_train_ = dm_inputs_train_[batch_id,:,:,0]
    dm_inputs_train_ = np.stack([dm_inputs_train_,dm_inputs_train_,dm_inputs_train_], axis=-1)

    CAM_hm_train_ = CAM_hm_train_
    hm_add_train_ = hm_add_train_[batch_id,:,:,:,:]
    valid_hm_train_ = valid_hm_train_[:,batch_id,:,:,:]

    # 绘制深度图
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1,1,1)
    # ax.axis('off')
    im = ax.imshow(dm_inputs_train_[:, :, :])
    plt.savefig(base_file_dir_im + file_namer_im + '\\' + 'image_.png')
    plt.clf()
    plt.close(fig)

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    # ax.axis('off')
    ax1.imshow(CAM_hm_train_[:, :, 0])
    ax2.imshow(CAM_hm_train_[:, :, 1])
    plt.savefig(base_file_dir_im + file_namer_im + '\\' + 'image_CAM.png')
    plt.clf()
    plt.close(fig)

    # 绘制热度图
    for i in range(14):
        fig = plt.figure(figsize=(18, 12))
        ax_L = [fig.add_subplot(2, 3, j + 1) for j in range(6)]
        # ax.axis('off')
        ax_L[0].imshow(valid_hm_train_[0, :, :, i])
        ax_L[1].imshow(valid_hm_train_[1, :, :, i])
        ax_L[3].imshow(valid_hm_train_[2, :, :, i])
        ax_L[4].imshow(valid_hm_train_[3, :, :, i])

        ax_L[2].imshow(hm_add_train_[:, :, i, 0])
        ax_L[5].imshow(hm_add_train_[:, :, i, 1])



        plt.savefig(base_file_dir_im + file_namer_im + '\\' + 'image_hm_'+str(i)+'.png')
        plt.clf()
        plt.close(fig)


is_see_all = False
if is_see_all:
    for step in tqdm(range(10001,10201)):
        for sub_step in range(5):
            file_name  = str(step).zfill(7)+"_" +str(sub_step)
            setDir(base_file_dir_im + file_name)
            dm_inputs_train_ = np.load(file=base_file_dir + file_name + "dm_inputs_train_.npy")
            CAM_hm_train_ = np.load(file=base_file_dir + file_name + "CAM_hm_train_.npy")
            hm_add_train_ = np.load(file=base_file_dir + file_name + "hm_add_train_.npy")
            valid_hm_train_ = np.load(file=base_file_dir + file_name + "valid_hm_train_.npy")

            for i in range(25):
                fig = plt.figure()

                ax_L = [fig.add_subplot(6, 7, j+1) for j in range(42)]

                for j in range(7):
                    im = ax_L[j].imshow(valid_hm_train_[0,i,:,:,j], cmap=matplotlib.cm.jet)
                    ax_L[j].axis('off')
                    im = ax_L[j + 7].imshow(hm_add_train_[i,:,:,j,0], cmap=matplotlib.cm.jet)
                    ax_L[j + 7].axis('off')
                    im = ax_L[j + 7*2].imshow(valid_hm_train_[3,i,:,:,j], cmap=matplotlib.cm.jet)
                    ax_L[j + 7*2].axis('off')


                    im = ax_L[j + 7*3].imshow(valid_hm_train_[0,i,:,:,j+7], cmap=matplotlib.cm.jet)
                    ax_L[j + 7*3].axis('off')
                    im = ax_L[j + 7*4].imshow(hm_add_train_[i,:,:,j+7,0], cmap=matplotlib.cm.jet)
                    ax_L[j + 7*4].axis('off')
                    im = ax_L[j + 7*5].imshow(valid_hm_train_[3,i,:,:,j+7], cmap=matplotlib.cm.jet)
                    ax_L[j + 7*5].axis('off')

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)

                plt.savefig(base_file_dir_im + file_name+'\\'+str(i).zfill(5)+'.png')
                plt.clf()
                plt.close(fig)


                fig = plt.figure()
                ax_I = [fig.add_subplot(1, 3, j+1) for j in range(3)]

                im = ax_I[0].imshow(dm_inputs_train_[i,:,:,0], cmap=matplotlib.cm.jet)
                ax_I[0].axis('off')
                im = ax_I[1].imshow(CAM_hm_train_[:,:,0], cmap=matplotlib.cm.jet)
                ax_I[1].axis('off')
                im = ax_I[2].imshow(CAM_hm_train_[:,:,1], cmap=matplotlib.cm.jet)
                ax_I[2].axis('off')

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)

                plt.savefig(base_file_dir_im + file_name+'\\'+str(i).zfill(5)+'_.png')
                plt.clf()
                plt.close(fig)