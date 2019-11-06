from __future__ import print_function, absolute_import, division

#import gpu_config
import tensorflow as tf
from network.slim import scopes, ops, losses, variables

FLAGS = tf.app.flags.FLAGS

_batch_norm_params={'decay':0.99,
                    'epsilon':0.001,
                    'center':True,
                    'scale':True}

# simultaneously regressing the 3D offset and the 2D heatmap
# v0 + dropout on the fully connected layer
TOWER_NAME = 'um_v1'
CAM_num = 0
def _residual(ins, num_out=None):
    ''' the bottleneck residual module
    Args:
        ins: the inputs
        k: kernel size
        num_out: number of the output feature maps, default set as the same as input
    Returns:
        residual network output
    '''
    num_in = ins.shape[-1].value
    if num_out is None:
        num_out = num_in

    with scopes.arg_scope([ops.conv2d],
                         stddev=0.01,
                         activation=tf.nn.relu,
                         batch_norm_params=_batch_norm_params,
                         weight_decay=0.0005,
                         stride=1,
                         padding='SAME'):
        half_num_in = int(num_in//2)
        out_1 = ops.conv2d(ins, half_num_in, [1,1])
        k = FLAGS.kernel_size
        out_1 = ops.conv2d(out_1, half_num_in, [k,k])
        out_1 = ops.conv2d(out_1, num_out, [1,1])

        if num_out == num_in:
            out_2 = ins
        else:
            out_2 = ops.conv2d(ins, num_out, [1,1])
        return out_1+out_2

MID_FEA_MAP = None
def _hourglass(ins, n):
    ''' hourglass is created recursively, each time the module spatial resolution remains the same
    '''
    upper1 = _residual(ins)
    
    k = FLAGS.kernel_size
    lower1 = ops.max_pool(ins, [k,k], stride=2, padding='SAME')
    lower1 = _residual(lower1)

    if n > 1:
        lower2 = _hourglass(lower1, n-1)
    else:
        lower2 = lower1

    lower3 = _residual(lower2)
    upper2 = ops.upsampling_nearest(lower3, 2)
    print('[hourglass] n={}, shape={}'.format(n, upper1.shape))

    return upper1+upper2

def detect_net(dm_inputs, cfgs, coms, num_jnt, is_training=True, scope=''):
    end_points = {}
    end_points['hm_outs'] = []
    end_points['hm3_outs'] = []
    end_points['um_outs'] = []

    end_points['hm_CAM1'] = []
    end_points['hm_CAM2'] = []
    end_points['hm_add1'] = []
    end_points['hm_add2'] = []

    end_points['dm_inputs'] = dm_inputs

    global CAM_num
    CAM_num = 0
    with tf.name_scope(scope, 'hg_net'):
        with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                              is_training = is_training):
            
            input_w, input_h = dm_inputs.shape[2].value, dm_inputs.shape[1].value

            # initial image processing (from 512*512 -> 128*128)
            with tf.variable_scope('hg_imgproc'):
                # 512*512 -> 256*256
                conv_1 = ops.conv2d(dm_inputs, 32, [7,7], stride=2, padding='SAME',
                                   batch_norm_params=_batch_norm_params, weight_decay=0.0005)
                conv_2 = _residual(conv_1, 64)

                # 256*256 -> 128*128
                pool_1 = ops.max_pool(conv_2, kernel_size=2, stride=2, padding='SAME')
                conv_3 = _residual(pool_1) 
                conv_4 = _residual(conv_3, FLAGS.num_fea) 
                hg_ins = conv_4

                global MID_FEA_MAP
                MID_FEA_MAP = hg_ins

            if input_w == 512 and input_h == 512:
                num_resize = 6
            elif input_w == 256 and input_h == 256:
                num_resize = 5
            elif input_w == 128 and input_h == 128:
                num_resize = 4

            else:
                raise ValueError('unknown input depth map shape')

            output_w, output_h = int(input_w/4), int(input_h/4)
            batch_size = tf.shape(dm_inputs)[0]
            tiny_dm = tf.image.resize_images(dm_inputs, (output_h, output_w), 2)

            uu, vv = tf.meshgrid(tf.range(output_h), tf.range(output_w))
            uu, vv = tf.cast(uu, tf.float32), tf.cast(vv, tf.float32)
            uu = tf.expand_dims(tf.divide(uu, float(output_w/2)) - 1.0, axis=-1)
            vv = tf.expand_dims(tf.divide(vv, float(output_h/2)) - 1.0, axis=-1)
            uu = tf.expand_dims(uu, axis=0)
            vv = tf.expand_dims(vv, axis=0)
            uu = tf.tile(uu, [batch_size, 1, 1, 1])
            vv = tf.tile(vv, [batch_size, 1, 1, 1])
            uvd = tf.concat([uu,vv,tiny_dm], axis=-1) 

            # the hour glass
            for i in range(FLAGS.num_stack):
                hg_outs = _hourglass(hg_ins, n=num_resize)

                ll = _residual(hg_outs)
                ll = ops.conv2d(ll, FLAGS.num_fea, [1,1], stride=1, padding='SAME',
                                activation=tf.nn.relu,
                                batch_norm_params=_batch_norm_params,
                                weight_decay=0.0005)
                
                hm_out = ops.conv2d(ll, num_jnt, [1,1], stride=1, padding='SAME',
                                    activation=None,
                                    weight_decay=0.0005)
                
                hm3_in = tf.concat([ll, uvd], axis=-1)
                hm3_in = _residual(hm3_in, 128)
                hm3_out = ops.conv2d(hm3_in, num_jnt, [1,1], stride=1, padding='SAME',
                                    activation=None,
                                    weight_decay=0.0005)

                um_in = tf.concat([hg_outs, hm_out, hm3_out], axis=-1)
                um_in = _residual(_residual(um_in, 256))

                um_in_mask = tf.concat([hg_outs, hm_out, hm3_out], axis=-1)
                mask = tf.tile(tf.less(tiny_dm, -0.9), (1,1,1,um_in_mask.get_shape()[-1].value))
                um_in_mask = tf.where(mask, tf.zeros_like(um_in_mask), um_in_mask)
                um_in_mask = _residual(_residual(um_in_mask, 256))

                um_in_comb = tf.concat([um_in, um_in_mask], axis=-1)
                um_in_comb = _residual(um_in_comb)
                um_in_comb = tf.concat([um_in_comb, uvd], axis=-1)

                um_full = ops.conv2d(um_in_comb, 512, [1,1], stride=1, padding='SAME',
                                         activation=tf.nn.relu,
                                         batch_norm_params=None,
                                         weight_decay=0.0005)
                um_full = ops.dropout(um_full)
                um_full = ops.conv2d(um_full, 512, [1,1], stride=1, padding='SAME',
                                         activation=tf.nn.relu,
                                         batch_norm_params=None,
                                         weight_decay=0.0005)
                um_full = ops.dropout(um_full)

                um_out = ops.conv2d(um_full, num_jnt*3, [1,1], stride=1, padding='SAME',
                                     activation=None,
                                     batch_norm_params=None,
                                     weight_decay=0.0005)
                end_points['hm_outs'].append(hm_out)
                end_points['hm3_outs'].append(hm3_out)
                end_points['um_outs'].append(um_out)

                # 用图网络，对[hm_out, hm3_out, um_out进行更改
                cur_node_representations = hm_out
                CAM_num = CAM_num + 1
                if False:
                    mask_2d = tf.cast(tf.less(tiny_dm, -0.9), dtype=tf.float32)
                with tf.variable_scope('GNN_' + str(i) + '_' + str(CAM_num)):

                    weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                    l2_regularizer = losses.l2_regularizer(0.0005)
                    learnable_Adj = variables.variable('learnable_adj_weights_' + str(CAM_num),
                                                       shape=[14, 14],
                                                       initializer=weights_initializer,
                                                       regularizer=l2_regularizer,
                                                       trainable=True,
                                                       restore=True)

                    cur_node_representations, add_hm = _apply_gnn_layer(
                        cur_node_representations,
                        learnable_Adj, uvd)
                    cur_node_representations = tf.contrib.layers.layer_norm(cur_node_representations)
                hm_out = cur_node_representations
                end_points['hm_outs'].append(hm_out)

                cur_node_representations = hm3_out
                CAM_num = CAM_num + 1
                with tf.variable_scope('GNN_' + str(i) + '_' + str(CAM_num)):

                    weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                    l2_regularizer = losses.l2_regularizer(0.0005)
                    learnable_Adj = variables.variable('learnable_adj_weights_' + str(CAM_num),
                                                       shape=[14, 14],
                                                       initializer=weights_initializer,
                                                       regularizer=l2_regularizer,
                                                       trainable=True,
                                                       restore=True)
                    if i==0:
                        end_points['hm_CAM1'] = learnable_Adj
                        end_points['hm_add1'] = add_hm

                    else:
                        end_points['hm_CAM2'] = learnable_Adj
                        end_points['hm_add2'] = add_hm
                    cur_node_representations, _ = _apply_gnn_layer(
                        cur_node_representations,
                        learnable_Adj, uvd)
                    cur_node_representations = tf.contrib.layers.layer_norm(cur_node_representations)
                hm3_out = cur_node_representations
                end_points['hm3_outs'].append(hm3_out)
                if False:
                    cur_node_representations = um_out
                    CAM_num = CAM_num + 1
                    with tf.variable_scope('GNN_' + str(i) + '_' + str(CAM_num)):

                        weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                        l2_regularizer = losses.l2_regularizer(0.0005)
                        learnable_Adj = variables.variable('learnable_adj_weights_' + str(CAM_num),
                                                           shape=[14*3, 14*3],
                                                           initializer=weights_initializer,
                                                           regularizer=l2_regularizer,
                                                           trainable=True,
                                                           restore=True)

                        cur_node_representations = _apply_gnn_layer(
                            cur_node_representations,
                            learnable_Adj, uvd)
                        cur_node_representations = tf.contrib.layers.layer_norm(cur_node_representations)
                    um_out = cur_node_representations
                    end_points['um_outs'].append(um_out)
                else:
                    with tf.variable_scope('GNN_no_local_um' + str(i)):
                        um_out = no_local(um_out)
                    end_points['um_outs'].append(um_out)

                if i < FLAGS.num_stack-1:
                    tmp_out = tf.concat([hm_out, hm3_out, um_out], axis=-1)
                    tmp_out_reshaped = ops.conv2d(tmp_out, FLAGS.num_fea, [1,1], stride=1,
                                                 batch_norm_params=None, 
                                                 activation=None)
                    inter = ops.conv2d(ll, FLAGS.num_fea, [1,1], stride=1,
                                      batch_norm_params=None, 
                                      activation=None)
                    hg_ins = hg_ins + tmp_out_reshaped + inter

        return end_points

"""
hm_out_in = hm_out
hm3_out_in = hm3_out

output_list = []
# hm 3次
cur_node_representations = hm_out_in
last_residual_representations = tf.zeros_like(cur_node_representations)
for layer_idx in range(3):
    CAM_num = CAM_num + 1
    with tf.variable_scope('GNN_' + str(layer_idx) + '_' + str(CAM_num)):

        weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
        l2_regularizer = losses.l2_regularizer(0.0005)
        learnable_Adj = variables.variable('learnable_adj_weights_' + str(CAM_num),
                                           shape=[14, 14],
                                           initializer=weights_initializer,
                                           regularizer=l2_regularizer,
                                           trainable=True,
                                           restore=True)

        if layer_idx % 2 == 0:
            t = cur_node_representations
            if layer_idx > 0:
                cur_node_representations += last_residual_representations
                cur_node_representations /= 2
            last_residual_representations = t
        cur_node_representations = _apply_gnn_layer(
            cur_node_representations,
            learnable_Adj, uvd)
        cur_node_representations = tf.contrib.layers.layer_norm(cur_node_representations)

        if layer_idx == 1 or layer_idx == 2:
            output_list.append(cur_node_representations)

end_points['hm_outs_GNN2'].append(output_list[0])
end_points['hm_outs_GNN5'].append(output_list[1])

# hm3 2次
cur_node_representations = hm3_out_in
last_residual_representations = tf.zeros_like(cur_node_representations)
for layer_idx in range(2):
    CAM_num = CAM_num + 1
    with tf.variable_scope('GNN_' + str(layer_idx) + '_' + str(CAM_num)):

        weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
        l2_regularizer = losses.l2_regularizer(0.0005)
        learnable_Adj = variables.variable('learnable_adj_weights_' + str(CAM_num),
                                           shape=[14, 14],
                                           initializer=weights_initializer,
                                           regularizer=l2_regularizer,
                                           trainable=True,
                                           restore=True)

        if layer_idx % 2 == 0:
            t = cur_node_representations
            if layer_idx > 0:
                cur_node_representations += last_residual_representations
                cur_node_representations /= 2
            last_residual_representations = t
        cur_node_representations = _apply_gnn_layer(
            cur_node_representations,
            learnable_Adj, uvd)
        cur_node_representations = tf.contrib.layers.layer_norm(cur_node_representations)

        if layer_idx == 0 or layer_idx == 1:
            output_list.append(cur_node_representations)

end_points['hm3_outs_GNN2'].append(output_list[2])
end_points['hm3_outs_GNN5'].append(output_list[3])

# um 2次
with tf.variable_scope('GNN_no_local_um1'):
    um_out = no_local(um_out)
    output_list.append(um_out)
with tf.variable_scope('GNN_no_local_um2'):
    um_out = no_local(um_out)
    output_list.append(um_out)
end_points['um_outs_GNN2'].append(output_list[4])
end_points['um_outs_GNN5'].append(output_list[5])

return end_points
"""
def _apply_gnn_layer(node_embeddings_: tf.Tensor,
                      adjacency_lists: tf.Tensor,
                      uvd: tf.Tensor,
                      num_timesteps: int = 3,
                      gated_unit_type: str = "gru",
                      activation_function: str = "tanh",
                      message_aggregation_function: str = "sum"
                      ) -> tf.Tensor:


    #cur_node_states = tf.reshape(node_embeddings,(batch, -1))
    n, h, w, c = node_embeddings_.shape
    node_embeddings = tf.reshape(node_embeddings_,(-1,c))
    node_embeddings = tf.matmul(node_embeddings, adjacency_lists)  # [D*B, G]
    node_embeddings = tf.reshape(node_embeddings, (n, h, w, c))
    node_embeddings_ = node_embeddings

    for i in range(num_timesteps):
        with tf.variable_scope('GNN_num_timesteps_' + str(i)):
            node_embeddings = no_local_one_time(node_embeddings)

    node_embeddings = tf.concat([node_embeddings,node_embeddings_, uvd], axis=-1)
    # === Prepare things we need across all timesteps:
    node_embeddings = _residual(node_embeddings, num_out=c)

    return node_embeddings, node_embeddings_
def no_local(X):
    s = X.get_shape().as_list()
    step1_out_chan = s[-1] * 2
    X_1_B = ops.conv2d(X, step1_out_chan, [1, 1], stride=1, padding='SAME',
               activation=None,
               batch_norm_params=None,
               weight_decay=0.0005)
    X_2_B = ops.conv2d(X, step1_out_chan, [1, 1], stride=1, padding='SAME',
               activation=None,
               batch_norm_params=None,
               weight_decay=0.0005)
    X_3_B = ops.conv2d(X, step1_out_chan, [1, 1], stride=1, padding='SAME',
               activation=None,
               batch_norm_params=None,
               weight_decay=0.0005)

    output_list = []
    for batch_num in range(s[0]):
        #X_o = X[batch_num, :, :, :]
        X_1 = X_1_B[batch_num, :, :, :]
        X_2 = X_2_B[batch_num, :, :, :]
        X_3 = X_3_B[batch_num, :, :, :]

        X_1 = tf.reshape(X_1, (-1,step1_out_chan))
        X_2 = tf.reshape(X_2, (-1,step1_out_chan))
        X_3 = tf.reshape(X_3, (-1,step1_out_chan))

        X_2 = tf.transpose(X_2)

        X_12 = tf.matmul(X_1, X_2)
        X_12 = tf.nn.softmax(X_12)

        X_123 = tf.matmul(X_12, X_3)

        X_123 = tf.reshape(X_123, (s[1],s[2],step1_out_chan))

        output_list.append([X_123])
    X_123 = tf.squeeze(tf.stack(output_list))
    X_123 = ops.conv2d(X_123, s[-1], [1, 1], stride=1, padding='SAME',
               activation=None,
               batch_norm_params=None,
               weight_decay=0.0005)
    X_123 = X_123+X

    return X_123

def no_local_one_time(X):
    s = X.get_shape().as_list() #s[0,1,2,3]
    step1_out_chan = s[-1] * 2
    X_1 = ops.conv2d(X, step1_out_chan, [1, 1], stride=1, padding='SAME',
               activation=None,
               batch_norm_params=None,
               weight_decay=0.0005)
    X_2 = ops.conv2d(X, step1_out_chan, [1, 1], stride=1, padding='SAME',
               activation=None,
               batch_norm_params=None,
               weight_decay=0.0005)
    X_3 = ops.conv2d(X, step1_out_chan, [1, 1], stride=1, padding='SAME',
               activation=None,
               batch_norm_params=None,
               weight_decay=0.0005)

    # 将 n h w 转换成
    X_1 = tf.reshape(X_1,(s[0],s[1]*s[2],step1_out_chan))
    X_2 = tf.reshape(X_2,(s[0],s[1]*s[2],step1_out_chan))
    X_2 = tf.transpose(X_2, [0, 2, 1])
    X_3 = tf.reshape(X_3,(s[0],s[1]*s[2],step1_out_chan))

    X_12 = tf.matmul(X_1, X_2)
    X_12 = tf.nn.softmax(X_12)

    X_123 = tf.matmul(X_12, X_3)
    X_123 = tf.reshape(X_123, (s[0],s[1], s[2], step1_out_chan))

    X_123 = ops.conv2d(X_123, s[-1], [1, 1], stride=1, padding='SAME',
               activation=None,
               batch_norm_params=None,
               weight_decay=0.0005)
    X_123 = X_123+X

    return X_123
