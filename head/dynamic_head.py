import tensorflow as tf
import numpy as np
from common.util import ConvBlock_gn, Conv2d
from postprocess import *

DEBUG = False

def points_nms(heat, kernel = 2):
    padding = [[0,0], [1,1], [1,1], [0,0]]
    hmax = tf.nn.max_pool2d(tf.pad(heat, padding),(kernel,kernel), strides=1, padding='VALID')
    keep = tf.cast((hmax[:,:-1,:-1,:] == heat),tf.float32)
    return keep * heat

@tf.function
def dice_loss(input, target):
    channel = input.shape[-1]
    input = tf.reshape(input,[-1, channel])
    target = tf.cast(tf.reshape(target,[-1, channel]),tf.float32)

    a = tf.reduce_sum(input * target, axis = 0)
    b = tf.reduce_sum(input * input, axis = 0) + 1e-6
    c = tf.reduce_sum(target * target, axis = 0) + 1e-6
    d = (2 * a) / (b + c)
    return 1. - d

@tf.function
def sigmoid_focal_loss(pred,
                          target,
                          weight=1.0,
                          gamma=2.0,
                          alpha=0.25,
                          avg_factor=None):
    pred_sigmoid = tf.sigmoid(pred)
    target = tf.cast(target, tf.float32)

    fl =  -1 * ( alpha * target * 1.0 * tf.math.pow(1 - pred_sigmoid, gamma) * tf.math.log( pred_sigmoid + 1e-9 ) + \
        (1 - alpha) * (1 - target) * 1.0* tf.math.pow(pred_sigmoid, gamma) * tf.math.log( 1 - pred_sigmoid + 1e-9 )
        ) 

    loss = weight * tf.reduce_sum(fl) / avg_factor

    return loss

class DynamicHead(tf.keras.Model):
    def __init__(
                self, 
                cfg,
                **kwargs):
        super(DynamicHead,self).__init__(**kwargs)
        self.stacked_convs = cfg["stacked_convs"]
        self.in_channel = cfg["in_channels"]
        self.out_channel = cfg["seg_feat_channels"]
        self.num_grids = cfg["num_grids"]
        self.num_classes = cfg["num_classes"]
        self.scale_ranges = cfg["scale_ranges"]
        # self.scale_ranges = [(1, 32), (16, 64), (32, 128), (64, 256), (128, 512)]
        self.sigma = cfg["sigma"]
        self.strides = cfg["strides"]
        self.cate_conv_list = []
        self.kernel_conv_list = []
        self.upsample_list = []
        self.unified_conv_list = []
        self.unified_final_conv = []
        self.init_layers()

    def init_layers(self):
        ### upsample layers
        for i in range(4): ### [1,2,3]
            tmp_list = []
            for j in range(i):
                tmp_list.append(
                    tf.keras.layers.UpSampling2D(
                        size=(2,2),interpolation = 'bilinear', name = "mask_branch/upsample_" + str(i+1)
                    )
                )
            self.upsample_list.append(tmp_list)
        ###conv layers
        for i in range(4):
            if i == 3:
                inchannel = self.in_channel + 2
            else:
                inchannel = self.in_channel
            tmp_list = []
            if i == 0:
                tmp_list.append(
                    ConvBlock_gn(
                            inchannel, self.out_channel,3,1,1, name = "mask_branch/P{}/layer{}".format(i+2, 1)
                    )
                )
            for j in range(i):
                if j == 0:
                    tmp_list.append(
                        ConvBlock_gn(
                                inchannel, self.out_channel,3,1,1, name = "mask_branch/P{}/layer{}".format(i+2, j+1)
                        )
                    )
                else:
                    tmp_list.append(
                        ConvBlock_gn(
                                self.out_channel, self.out_channel,3,1,1, name = "mask_branch/P{}/layer{}".format(i+2, j+1)
                        )
                    )
            self.unified_conv_list.append(tmp_list)
        

        for i in range(self.stacked_convs):
            chn = self.in_channel + 2 if i == 0 else self.out_channel
            self.kernel_conv_list.append(
                ConvBlock_gn(chn,self.out_channel,3,1,1,name = "kernel_branch/conv_{}".format(i))
                )

            chn = self.in_channel if i == 0 else self.out_channel
            self.cate_conv_list.append(
                ConvBlock_gn(chn,self.out_channel,3,1,1,name = "cate_branch/conv_{}".format(i))
                )

        # for num_grid in self.num_grids:
        #     self.solo_ins_list.append(
        #         Conv2d(
        #             self.out_channel, num_grid**2,1
        #             )
        #     )

        # self.solo_cate = Conv2d(
        #     self.out_channel,self.num_classes,3,1,1, 
        #     name = "cate_branch/conv_{}".format(self.stacked_convs)
        #     )
        self.solo_cate = tf.keras.layers.Conv2D(self.num_classes,
                    (3,3),
                    padding = "SAME",  
                    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer = tf.constant_initializer(value = -3.0 ), #-np.log(99.)
                    name = "cate_branch/conv_{}".format(self.stacked_convs)
                )
        # self.solo_kernel = Conv2d(
        #     self.out_channel,self.out_channel,1,1,0, 
        #     name = "kernel_branch/conv_{}".format(self.stacked_convs)
        #     )
        self.solo_kernel = tf.keras.layers.Conv2D(self.out_channel,
                    (3,3),
                    padding = "SAME", 
                    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer = tf.constant_initializer(value = -0.1), #-np.log(99.)
                    name = "cate_branch/conv_{}".format(self.stacked_convs)
                )
        ### add 1x1 at the end
        self.solo_mask = ConvBlock_gn(
                self.out_channel,self.out_channel,1,1,0, name = "mask_branch/final_layer"
            )

    def coord(self, batch, height, width):
        x_range = np.linspace(-1.0, 1.0, width)
        y_range = np.linspace(-1.0, 1.0, height)
        x, y = np.meshgrid(x_range, y_range)
        y = np.repeat(np.expand_dims(y, axis = 0), batch, axis = 0)
        x = np.repeat(np.expand_dims(x, axis = 0), batch, axis = 0)
        y = np.expand_dims(y, axis = -1)
        x = np.expand_dims(x, axis = -1)
        coord_feat = np.concatenate([y, x], axis = -1)
        return coord_feat

    def unified_mask_branch(self, features):### 结构还需要调整
        
        num_features = 4
        feat_list = []
        for i in range(num_features):
            if i == num_features - 1:
                batch, height, width, _ = features[i].shape
                if batch is None:
                    batch = 1
                coord_feat = tf.constant(self.coord(batch, height, width), dtype=tf.float32)
                feat = tf.concat([features[i], coord_feat], -1)
            else:
                feat = features[i]
            if i == 0:
                feat = self.unified_conv_list[i][0](feat)
                feat_list.append(feat)
                continue
            for j in range(i):
                feat = self.unified_conv_list[i][j](feat)
                feat = self.upsample_list[i][j](feat)
            feat_list.append(feat)
        _, h, w, c = feat_list[0].shape
        unified_feat = feat_list[0]
        for i in range(1, len(feat_list)):
            feat = feat_list[i]
            if feat.shape[1] != h or feat.shape[2] != w:
                unified_feat += feat[:, :h, :w, :]
            else:
                unified_feat += feat
            
        # unified_feat = tf.math.add_n(feat_list, name="unified_mask/unified_features")

        output = self.solo_mask(unified_feat)
        return output

    def split_feats(self, features):
        res = []
        num_feature = len(features)
        for _index in range(num_feature):
            if _index == 0:
                _, h, w, _ = features[_index].shape
                res.append( tf.image.resize(features[_index], [h//2, w//2], method='bilinear') )
            elif _index == num_feature - 1:
                _, h, w, _ = features[_index - 1].shape
                res.append( tf.image.resize(features[-1], [h, w], method='bilinear') )
            else:
                res.append(features[_index])
        return res
        
    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        kernel_feat = x
        cate_feat = x
        # cate branch
        for i, cate_layer in enumerate(self.cate_conv_list):
            if i == 0:
                num_grid = self.num_grids[idx]
                # print(num_grid)
                cate_feat = tf.image.resize(cate_feat,[num_grid,num_grid],method='bilinear')
                # print(cate_feat.shape)
            cate_feat = cate_layer(cate_feat)

        cate_pred = self.solo_cate(cate_feat)

        # ins branch
        # concat coord
        batch, height, width, _ = kernel_feat.shape
        if batch is None:
            batch = 1
        coord_feat = tf.constant(self.coord(batch, height, width), dtype=tf.float32)
        feat = tf.concat([kernel_feat, coord_feat], -1)
        # ins_feat = tf.concat([ins_feat, coord_feat], -1)
        for i, kernel_conv in enumerate(self.kernel_conv_list):
            if i ==0 :
                num_grid = self.num_grids[idx]
                feat = tf.image.resize(feat, [num_grid, num_grid], method='bilinear')
            feat = kernel_conv(feat)

        kernel_pred = self.solo_kernel(feat)
        if eval:
            cate_pred = points_nms(tf.sigmoid(cate_pred), kernel=2)   
        return kernel_pred, cate_pred

    def call(self, input, eval = False, training=True):
        new_feats = self.split_feats(input)

        featmap_size = [(featmap.shape[1], featmap.shape[2]) for featmap in new_feats]
        kernel_preds = []
        cate_preds = []
        ins_preds = []
        
        for i,feat in enumerate(new_feats):
            kernel_pred, cate_pred = self.forward_single(feat,i)
            kernel_preds.append(kernel_pred)
            cate_preds.append(cate_pred)

        feature_pred = self.unified_mask_branch(input)

        batch, height, width, channel = feature_pred.shape

        # feature_pred = tf.reshape(feature_pred, [1, batch * channel, height, width ])

        for i, kernel_pred in enumerate(kernel_preds): ### kernel_pred shape: maybe [N, 40, 40, 256]
            ins_pred_batch = []
            if kernel_pred.shape[0] == 1 or feature_pred.shape[0] == 1:
                kernel = tf.reshape(kernel_pred, [-1, channel])
                kernel = tf.transpose(kernel)
                kernel = tf.reshape(kernel, [1, 1, channel, -1])
                ins = tf.nn.conv2d(feature_pred, kernel, strides = 1, padding="SAME")
            else:
                for single_kernel_pred, single_feature_pred in zip( kernel_pred, feature_pred): ### 遍历batch
                    kernel = tf.reshape(single_kernel_pred, [-1, channel])
                    kernel = tf.transpose(kernel)  ### [channel, -1]
                    kernel = tf.reshape(kernel, [1, 1, channel, -1])
                    # kernel = tf.reshape(single_kernel_pred, [1, 1, channel, -1])
                    feature = tf.expand_dims(single_feature_pred, axis = 0)
                    ins_pred = tf.nn.conv2d(feature, kernel, strides = 1, padding="SAME")
                    ins_pred_batch.append(ins_pred)
                ins = tf.concat(ins_pred_batch, axis = 0) ### batch
            if eval:
                ins = tf.sigmoid(ins)
            else:
                ins = tf.image.resize(ins, featmap_size[i], method='bilinear')
            ins_preds.append(ins)
        return ins_preds, cate_preds

    def ground_truth(self, label, feature_size_list):
        # gt_areas = label['area']
        cate_gts = []
        mask_gts = []
        index_gts = []
        for (lower_bound, upper_bound),num_grid, feature_size in zip(self.scale_ranges, self.num_grids, feature_size_list): 
            mask_gt_batch = []
            cate_gt_batch = []
            index_gt_batch = []

            ### 在batch中遍历
            for area, mask, cate, center_h, center_w, ymin, xmin, ymax, xmax  \
                in zip(label['area'], label["mask"], label["category"], label["ycenter"], label["xcenter"], \
                    label['ymin'], label['xmin'], label['ymax'], label['xmax']):
                ### 为每种grid初始化ground truth
                mask_label = np.zeros([1, feature_size[0], feature_size[1], num_grid**2], dtype = np.int64) ## 实例gt,最原始的, [h,w, s^2]
                cate_label = np.zeros([1, num_grid, num_grid, self.num_classes], dtype = np.int64)                            ## grid格子，值为类别，[s,s]
                index_label = np.zeros([1, num_grid**2], dtype = np.bool)                                ## s*s格子，bool型，表示有没有实例  

                hit_indices = tf.squeeze(tf.where((area >= lower_bound) & (area <= upper_bound)), axis=-1)

                if len(hit_indices) == 0:
                    cate_gt_batch.append(cate_label)
                    mask_gt_batch.append(mask_label)
                    index_gt_batch.append(index_label)
                    continue
                
                half_hs = [0.5 * (_ymax - _ymin) * self.sigma for _ymax, _ymin in zip(ymax, ymin)]
                half_ws = [0.5 * (_xmax - _xmin) * self.sigma for _xmax, _xmin in zip(xmax, xmin)]
                
                for i in hit_indices.numpy():
                    # print("cate: ", cate[i])
                    ### 中心坐标归一化后在网格中真实坐标
                    coord_h = int( center_h[i] * num_grid )
                    coord_w = int( center_w[i] * num_grid )

                    ### 论文3.3.1引入的gt box
                    top_box = max(0, int( (center_h[i] - half_hs[i]) * num_grid) )
                    down_box = min(num_grid - 1, int( (center_h[i] + half_hs[i]) * num_grid ) )
                    left_box = max(0, int( (center_w[i] - half_ws[i]) * num_grid ) )
                    right_box  = min(num_grid - 1, int( (center_w[i] + half_ws[i]) * num_grid ) )

                    ###与中心点3x3区域取交
                    top = max(top_box, coord_h - 1)
                    down = min(down_box, coord_h + 1)
                    left = max(left_box, coord_w - 1)
                    right = min(right_box, coord_w + 1)

                    cate_label[0, top : down + 1, left : right + 1, cate[i] - 1] = 1

                    ### 将gt_mask resize到实际大小，与feature map 大小一致
                    mask_resize = tf.image.resize(tf.expand_dims(mask[i],-1), feature_size, method='bilinear')
                    mask_resize = tf.cast(mask_resize + 0.5, tf.int64)
                    for i in range(top, down + 1):
                        for j in range(left, right + 1):
                            index = int(i * num_grid + j)
                            mask_label[0, :, :, index] = mask_resize[:,:,0]
                            index_label[0, index] = True
                cate_gt_batch.append(cate_label)
                mask_gt_batch.append(mask_label)
                index_gt_batch.append(index_label)
            cate_gts.append(tf.concat(cate_gt_batch,axis = 0))
            mask_gts.append(tf.concat(mask_gt_batch,axis = 0))
            index_gts.append(tf.concat(index_gt_batch,axis = 0))
        return mask_gts, cate_gts, index_gts

    def calc_loss(self, preds, label):

        ins_preds, cate_preds = preds

        feature_size_list = [ featmap.shape.as_list()[1:3] for featmap in ins_preds]
        mask_gts, cate_gts, index_gts = self.ground_truth(label, feature_size_list)

        if 1:
            for id in range(len(cate_preds)):
                cate_pred = cate_preds[id]
                # cate_gt = cate_gts[id]
                print(
                    "cate pred num: {}, max: {} ".format( 
                        tf.reduce_sum(tf.cast(cate_pred > 0, tf.float32)).numpy(), 
                        # tf.reduce_sum(tf.cast(cate_pred <= 0, tf.float32)).numpy()
                        tf.reduce_max(cate_pred)
                        ) 
                    )
                # print(
                #     "cate gt max: {}, min: {} ".format( 
                #         tf.reduce_sum(tf.cast(cate_gt > 0, tf.float32)).numpy(), 
                #         # tf.reduce_sum(tf.cast(cate_gt <= 0, tf.float32)).numpy()
                #         tf.reduce_max(cate_gt)
                #         ) 
                #     )
        if 1:
            for id in range(len(ins_preds)):
                ins_pred = ins_preds[id]
                print(
                    "mask (pred > 0): (<=0) {}, max: {} , min: {} ".format( 
                        tf.reduce_sum(tf.cast(ins_pred > 0, tf.float32)).numpy() / (tf.reduce_sum(tf.cast(ins_pred <= 0, tf.float32)).numpy() + 1), 
                        tf.reduce_max(ins_pred), tf.reduce_min(ins_pred)
                        ) 
                    )

        # for index in index_gts:
        #     print("index shape: ", index.shape)
        ### ground truth instance
        ins_labels_gt = []
        for mask_batch, index_batch in zip(mask_gts, index_gts):
            ins_labels = []
            for mask_label, index_label in zip( mask_batch, index_batch ):
                # mask_label_transpose = tf.transpose(mask_label,(2,0,1))
                # ins_label = tf.transpose(tf.boolean_mask(mask_label_transpose, index_label),(1,2,0))
                ins_label = tf.boolean_mask(mask_label, index_label, axis = 2)
                # print("ins_label shape: ", ins_label.shape)
                ins_labels.append(ins_label)
            ins_labels_gt.append(tf.concat(ins_labels,axis = -1))

        # ### pred instance
        ins_labels_pred = []
        for mask_batch, index_batch in zip(ins_preds, index_gts):
            ins_labels = []
            for mask_label, index_label in zip( mask_batch, index_batch ):
                # mask_label_transpose = tf.transpose(mask_label,(2,0,1))
                # ins_label = tf.transpose(tf.boolean_mask(mask_label_transpose, index_label),(1,2,0))
                ins_label = tf.boolean_mask(mask_label, index_label, axis = 2)
                ins_labels.append(ins_label)
            ins_labels_pred.append(tf.concat(ins_labels,axis = -1))

        ins_index_gt = []
        for index_batch in zip(index_gts):
            index_labels = []
            for ins_ind in zip(index_batch):
                index_labels.append(tf.reshape(ins_ind, [-1]))
            ins_index_gt.append(tf.concat(index_labels,axis = 0))
        
        flatten_ins_index_gt = tf.concat(ins_index_gt, axis = 0)

        num_instance = tf.reduce_sum(tf.cast(flatten_ins_index_gt, tf.float32))

        ### dice loss
        loss_ins = []
        for ins_pred, ins_gt  in zip(ins_labels_pred, ins_labels_gt):
            # print("ins_pred shape: ", ins_pred.shape)
            # print("ins_gt shape: ", ins_gt.shape)
            if ins_pred.shape[2] == 0:
                continue
            # print("pred before sigmoid min: {}, max: {}".format(tf.reduce_min(ins_pred), tf.reduce_max(ins_pred)))
            pred = tf.math.sigmoid(ins_pred)
            # print("pred min: {}, max: {}".format(tf.reduce_min(pred), tf.reduce_max(pred)))
            # print("gt min: {}, max: {}".format(tf.reduce_min(ins_gt), tf.reduce_max(ins_gt)))
            loss_ins.append(dice_loss(pred,ins_gt))
        # print("len(loss_ins): ", len(loss_ins))
        loss_ins = tf.reduce_mean(tf.concat(loss_ins,0)) * 3.0

        ### cate
        cate_gt_list = []
        for cate_gt_batch in zip(cate_gts):   ### cate_batch's shape maybe  [batch_size,40,40,num_classes]
            # cate_list = []
            # for cate in zip(cate_batch):
            #     cate_list.append(tf.reshape(cate,[-1, self.num_classes]))
            cate_gt_reshape = tf.reshape(cate_gt_batch,[-1, self.num_classes])
            cate_gt_list.append(cate_gt_reshape)

        flatten_cate_gt = tf.concat(cate_gt_list, axis = 0)

        cate_preds_list = []
        for cate_pred_batch in cate_preds:  ### cate_pred_batch's shape maybe [batch_Size,40,40,num_classes]
            cate_pred_reshape = tf.reshape(cate_pred_batch,[-1, self.num_classes])
            cate_preds_list.append(cate_pred_reshape)
        
        flatten_cate_preds = tf.concat(cate_preds_list, axis = 0)

        ### loss cate here
        loss_cate = sigmoid_focal_loss(flatten_cate_preds, flatten_cate_gt, avg_factor=num_instance+1)

        return dict(loss_cate=loss_cate,loss_ins=loss_ins)

    def get_seg(self, all_preds, img_metas, cfg, rescale = None):

        seg_preds, cate_preds = all_preds
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].shape[1:3]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                tf.reshape(cate_preds[i][img_id],[-1, self.num_classes]) for i in range(num_levels)
            ]
            seg_pred_list = [
                seg_preds[i][img_id] for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = tf.concat(cate_pred_list, axis = 0)
            seg_pred_list = tf.concat(seg_pred_list, axis = -1)
            if DEBUG:
                print("cate_pred_list shape: {} ".format(cate_pred_list.shape))
                print("seg_pred_list_x shape: {} ".format(seg_pred_list.shape))
            result = self.get_seg_single(cate_pred_list, seg_pred_list, 
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list


    def get_seg_single(self,
                        cate_preds,
                        seg_preds,
                        featmap_size,
                        img_shape,
                        ori_shape,
                        scale_shape,
                        cfg,
                        rescale=False):
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)


        # # process.
        # inds = tf.where(cate_preds > cfg.score_thr)
        # # category scores.
        # # cate_scores = cate_preds[inds]
        # cate_scores = tf.gather_nd(cate_preds, inds)
        # if len(cate_scores) == 0:
        #     return None
        # # category labels.
        # cate_labels = inds[:, 1]

        # trans_size = tf.cumsum(tf.pow(self.num_grids,2))
        num_grids_square = [grid**2 for grid in self.num_grids]
        trans_size = np.cumsum(num_grids_square)
        # seg_size = np.cumsum(self.num_grids)

        n_stage = len(self.num_grids)
        # trans_diff_list = []
        # seg_diff_list = []
        # num_grids_list = []
        strides_list = []

        # trans_diff_list.append(tf.zeros(trans_size[0], dtype = tf.int64))
        # seg_diff_list.append(tf.zeros(trans_size[0], dtype = tf.int64))
        # num_grids_list.append(tf.ones(trans_size[0], dtype = tf.int64) * self.num_grids[0])
        strides_list.append(tf.ones(trans_size[0], dtype = tf.int64) * self.strides[0])

        for ind_ in range(1, n_stage):
            # trans_diff_list.append(tf.ones(num_grids_square[ind_], dtype = tf.int64) * trans_size[ind_ - 1])
            # seg_diff_list.append(tf.ones(num_grids_square[ind_], dtype = tf.int64) * seg_size[ind_ - 1])
            # num_grids_list.append(tf.ones(num_grids_square[ind_], dtype = tf.int64) * self.num_grids[ind_])
            strides_list.append(tf.ones(num_grids_square[ind_], dtype = tf.int64) * self.strides[ind_])

        # trans_diff = tf.concat(trans_diff_list, axis = 0)
        # seg_diff = tf.concat(seg_diff_list, axis = 0)
        # num_grids = tf.concat(num_grids_list, axis = 0)
        strides = tf.concat(strides_list, axis = 0)

        # process.
        inds = tf.where(cate_preds > cfg["score_thr"])
        cate_scores = tf.gather_nd(cate_preds, inds)
        if len(cate_scores) == 0:
            return None
        # trans_diff = tf.gather(trans_diff, inds[:, 0], axis=0)
        # seg_diff = tf.gather(seg_diff, inds[:, 0], axis=0)
        # num_grids = tf.gather(num_grids, inds[:, 0], axis=0)
        strides = tf.gather(strides, inds[:, 0], axis=0)
        
        cate_labels = inds[:, 1]
        seg_preds = tf.gather(seg_preds, inds[:, 0], axis=-1)
        seg_masks = seg_preds > cfg["mask_thr"]

        sum_masks = tf.reduce_sum(tf.cast(seg_preds > cfg["mask_thr"],tf.float32),axis=[0,1])

        # keep = sum_masks > strides  ### 一维向量
        keep = tf.squeeze(tf.where(sum_masks > tf.cast(strides,tf.float32)), axis = -1)
        if len(keep) == 0:
            return None
        seg_preds = tf.gather(seg_preds, keep, axis= -1)
        seg_masks = tf.gather(seg_masks, keep, axis= -1)
        sum_masks = tf.gather(sum_masks, keep, axis= -1)
        cate_scores = tf.gather(cate_scores, keep, axis= -1)
        cate_labels = tf.gather(cate_labels, keep, axis= -1)

        # mask scoring
        seg_score = tf.reduce_sum(seg_preds * tf.cast(seg_masks, tf.float32),axis=[0,1]) / sum_masks
        cate_scores *= seg_score

        # if len(cate_scores) == 0:
        #     return None

        ###----------------------------------------------------------------------------###
        # sort and keep top nms_pre
        sort_inds = tf.argsort(cate_scores, direction='DESCENDING')
        if len(sort_inds) > cfg["nms_pre"]:
            sort_inds = sort_inds[:cfg["nms_pre"]]
        seg_preds = tf.gather(seg_preds, sort_inds, axis=-1)
        seg_masks = tf.gather(seg_masks, sort_inds, axis=-1)
        sum_masks = tf.gather(sum_masks, sort_inds, axis=-1)
        cate_scores = tf.gather(cate_scores, sort_inds, axis=-1)      
        cate_labels = tf.gather(cate_labels, sort_inds, axis=-1)

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg["kernel"], sigma=cfg["sigma"], sum_masks=sum_masks)


        keep = tf.squeeze(tf.where(cate_scores >= cfg["update_thr"]), axis = -1)
        if len(keep) == 0:
            return None
        seg_preds = tf.gather(seg_preds, keep, axis = -1)
        cate_scores = tf.gather(cate_scores, keep, axis=-1)
        cate_labels = tf.gather(cate_labels, keep, axis=-1)

        # sort and keep top_k
        sort_inds = tf.argsort(cate_scores, direction='DESCENDING')
        if len(sort_inds) > cfg["max_per_img"]:
            sort_inds = sort_inds[:cfg["max_per_img"]]
        seg_preds = tf.gather(seg_preds, sort_inds, axis = -1)
        cate_scores = tf.gather(cate_scores, sort_inds, axis = -1)
        cate_labels = tf.gather(cate_labels, sort_inds, axis = -1)

        seg_preds = tf.image.resize(seg_preds, size = upsampled_size_out, method='bilinear')
        seg_masks = tf.image.resize(seg_preds[:h, :w, ...], ori_shape[:2], method='bilinear')
        seg_masks = seg_masks > cfg["mask_thr"]
        return seg_masks, cate_labels, cate_scores
