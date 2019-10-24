import numpy as np
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
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

valid_hms_all = np.load(
    file="F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\image\\hm\\data.npy")
print(valid_hms_all.shape)
# 变成4个30 32 32 的数组
#valid_hms_all = np.sum(valid_hms_all,axis=-1)
valid_hms_all = valid_hms_all[:,:,:,:,0]
print(valid_hms_all.shape)

for i in range(30):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis('off')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    ax1.imshow(valid_hms_all[0,i,:,:], cmap=matplotlib.cm.jet)
    ax2.imshow(valid_hms_all[1,i,:,:], cmap=matplotlib.cm.jet)
    ax3.imshow(valid_hms_all[2,i,:,:], cmap=matplotlib.cm.jet)
    im = ax4.imshow(valid_hms_all[3,i,:,:], cmap=matplotlib.cm.jet)

    #
    ax1.set_title( std_noise(valid_hms_all[0, i, :, :] ), fontsize=12, color='r')
    ax2.set_title( std_noise(valid_hms_all[1, i, :, :] ), fontsize=12, color='r')
    ax3.set_title( std_noise(valid_hms_all[2, i, :, :] ), fontsize=12, color='r')
    ax4.set_title( std_noise(valid_hms_all[3, i, :, :] ), fontsize=12, color='r')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig('F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\image\\hm\\'+str(i).zfill(5)+'.png')
    plt.clf()
    plt.close(fig)

