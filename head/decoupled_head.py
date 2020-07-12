import tensorflow as tf
from common.util import ConvBlock_gn, Conv2d
import tensorflow_addons as tfa
import numpy as np
from postprocess import *

DEBUG = False

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
    fl =  -1 * ( alpha * target * 1. * tf.math.pow(1 - pred_sigmoid, gamma) * tf.math.log( pred_sigmoid + 1e-6) + \
        (1 - alpha) * (1 - target) * 1.0 * tf.math.pow(pred_sigmoid, gamma) * tf.math.log( 1 - pred_sigmoid + 1e-6)
        )
    loss = weight * tf.reduce_sum(fl) / avg_factor
    return loss



def bce_loss(pred, target):
    loss = tf.keras.losses.binary_crossentropy(target, pred)
    return loss

def se_loss(pred, target, alpha = 0.001, avg_factor = None):
    pred_sigmoid = tf.sigmoid(pred)
    target = tf.cast(target, tf.float32)
    loss = tf.square(pred_sigmoid - target)
    loss = loss * target + alpha * (1 - target) * loss 
    loss = tf.reduce_sum(loss) / avg_factor
    return loss

def focal_loss(input, target, avg_factor, weight = 1.0):
    target = tf.cast(target, tf.float32)
    # return tf.reduce_sum(tfa.losses.sigmoid_focal_crossentropy(target, input)/avg_factor)
    return weight * tf.reduce_sum(tfa.losses.sigmoid_focal_crossentropy(target, input)) / avg_factor

def gather_axis(params, indices, axis=0):
    return tf.stack(tf.unstack(tf.gather(tf.unstack(params, axis=axis), indices)), axis=axis)

def points_nms(heat, kernel = 2):
    padding = [[0,0], [1,1], [1,1], [0,0]]
    hmax = tf.nn.max_pool2d(tf.pad(heat, padding),(kernel,kernel), strides=1, padding='VALID')
    keep = tf.cast((hmax[:,:-1,:-1,:] == heat),tf.float32)
    return keep * heat


class DecoupledHead(tf.keras.Model):
    def __init__(
                self, 
                cfg,
                **kwargs):
        super(DecoupledHead,self).__init__(**kwargs)
        self.stacked_convs = cfg["stacked_convs"]
        self.in_channel = cfg["in_channels"]
        self.out_channel = cfg["seg_feat_channels"]
        self.num_grids = cfg["num_grids"]
        self.num_classes = cfg["num_classes"]
        self.sigma = cfg["sigma"]
        self.strides = cfg["strides"]
        self.scale_ranges = cfg["scale_ranges"]
        # self.scale_ranges = [(1, 64), (32, 128), (64, 256), (128, 384),(256,512)]
        # self.scale_ranges = [(1, 96), (48, 160), (64, 256), (128, 384),(256,512)]
        self.cate_conv_list = []
        self.mask_conv_list_x = []
        self.mask_conv_list_y = []
        self.solo_ins_list_x = []
        self.solo_ins_list_y = []
        self.init_layers()

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

    def init_layers(self):
        
        for i in range(self.stacked_convs):
            chn = self.in_channel + 1 if i == 0 else self.out_channel
            self.mask_conv_list_x.append(
                ConvBlock_gn(chn,self.out_channel,3,1,1,name = "mask_branch/conv_x{}".format(i))
                # ConvBlock(chn,self.out_channel,3,1,1)
                )

            self.mask_conv_list_y.append(
                ConvBlock_gn(chn,self.out_channel,3,1,1, name = "mask_branch/conv_y{}".format(i))
                # ConvBlock(chn,self.out_channel,3,1,1)
                )

            chn = self.in_channel if i == 0 else self.out_channel
            self.cate_conv_list.append(
                ConvBlock_gn(chn,self.out_channel,3,1,1, name = "cate_branch/conv{}".format(i))
                # ConvBlock(chn,self.out_channel,3,1,1)
                )


        for num_grid in self.num_grids:
            self.solo_ins_list_y.append(
                # Conv2d(
                #     self.out_channel,num_grid,3,1,1, name = "mask_branch/grid_y{}".format(num_grid)
                #     )
                tf.keras.layers.Conv2D(num_grid,
                    (3,3),
                    padding = "SAME", 
                    # kernel_initializer='he_normal', 
                    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer = tf.constant_initializer(value = -np.log(99.)),
                    name = "mask_branch/grid_y{}".format(num_grid)
                )
            )
            self.solo_ins_list_x.append(
                # Conv2d(
                #     self.out_channel,num_grid,3,1,1, name = "mask_branch/grid_x{}".format(num_grid)
                #     )
                tf.keras.layers.Conv2D(num_grid,
                    (3,3),
                    padding = "SAME", 
                    # kernel_initializer='he_normal', 
                    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer = tf.constant_initializer(value = -np.log(99.)),
                    name = "mask_branch/grid_x{}".format(num_grid)
                )
            )
        self.solo_cate = tf.keras.layers.Conv2D(self.num_classes,
                    (3,3),
                    padding = "SAME", 
                    # kernel_initializer='he_normal', 
                    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer = tf.constant_initializer(value = -np.log(99.)),
                    name = "cate_branch/conv{}".format(self.stacked_convs)
                )
        # self.solo_cate = Conv2d(
        #     self.out_channel,self.num_classes,3,1,1, name = "cate_branch/conv{}".format(self.stacked_convs)
        #     )

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_feat = x
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
        # cate_pred = tf.nn.relu6(cate_pred) - 3

        # ins branch
        # concat coord
        batch, height, width, _ = ins_feat.shape

        def coord(batch, height, width):
            x_range = np.linspace(-1.0, 1.0, width)
            y_range = np.linspace(-1.0, 1.0, height)
            x, y = np.meshgrid(x_range, y_range)
            y = np.repeat(np.expand_dims(y, axis = 0), batch, axis = 0)
            x = np.repeat(np.expand_dims(x, axis = 0), batch, axis = 0)
            y = np.expand_dims(y, axis = -1)
            x = np.expand_dims(x, axis = -1)
            return y, x
        # ins_feat = tf.concat([ins_feat, coord_feat], -1)

        # _data = np.load("./coord_1200.npy", allow_pickle=True).item()
        # if eval:
        #     coord_feat = tf.constant(_data[height][0:1,...],dtype=tf.float32)
        # else:
        #     coord_feat = tf.constant(_data[height][0:2,...],dtype=tf.float32)
        if batch is None:
            batch = 1
        # coord_feat = tf.constant(coord(batch, height, width),dtype=tf.float32)
        # y, x = tf.split(coord_feat, 2, axis = -1)
        y, x = coord(batch, height, width)
        ins_feat_y = tf.concat([ins_feat, y], -1)
        ins_feat_x = tf.concat([ins_feat, x], -1)
        # for i, mask_conv in enumerate(self.mask_conv_list):
        #     ins_feat = mask_conv(ins_feat)
        for mask_conv_y, mask_conv_x in zip(self.mask_conv_list_y,self.mask_conv_list_x):
            ins_feat_y = mask_conv_y(ins_feat_y)
            ins_feat_x = mask_conv_x(ins_feat_x)

        _, h, w, _ = ins_feat_y.shape
        ins_feat_y = tf.image.resize(ins_feat_y, [h * 2, w * 2], method='bilinear')
        ins_feat_x = tf.image.resize(ins_feat_x, [h * 2, w * 2], method='bilinear')
        ins_pred_y = self.solo_ins_list_y[idx](ins_feat_y)
        ins_pred_x = self.solo_ins_list_x[idx](ins_feat_x)
        if eval:# to do
            # print("do eval")
            ins_pred_y = tf.image.resize(tf.sigmoid(ins_pred_y), upsampled_size, method='bilinear')
            ins_pred_x = tf.image.resize(tf.sigmoid(ins_pred_x), upsampled_size, method='bilinear')
            # print("cate pred max11:  ",tf.reduce_max(cate_pred))
            cate_pred = points_nms(tf.sigmoid(cate_pred), kernel=2)
        return ins_pred_y, ins_pred_x, cate_pred

    def call(self, input, eval = False, training=True):
        new_feats = self.split_feats(input)
        # shape_set = set()
        # for f in new_feats:
        #     shape_set.add(f.shape[2])
        # ins_preds, cate_preds = tf.map_fn(
        #     lambda x: self.forward_single(x[0],x[1]), (new_feats, np.array([0,1,2,3,4]))
        #     )
        

        ins_preds_y = []
        ins_preds_x = []
        cate_preds = []
        
        f_shape = new_feats[0].shape
        upsampled_size = [f_shape[1] * 2 ,f_shape[2] * 2]
        
        for i,feat in enumerate(new_feats):
            ins_pred_y, ins_pred_x, cate_pred = self.forward_single(feat, i, eval=eval, upsampled_size=upsampled_size)
            # print("cate pred max: {}, min: {} ".format( tf.reduce_sum(tf.cast(cate_pred > 0, tf.float32)).numpy(), \
            #     tf.reduce_sum(tf.cast(cate_pred <= 0, tf.float32)).numpy()
            #     ) 
            #     )
            ins_preds_y.append(ins_pred_y)
            ins_preds_x.append(ins_pred_x)
            cate_preds.append(cate_pred)

        return ins_preds_y, ins_preds_x, cate_preds

    def ground_truth(self, label, feature_size_list):
        # gt_areas = label['area']
        cate_gts = []
        mask_gts = []
        index_gts = []
        for (lower_bound, upper_bound),num_grid, feature_size in zip(self.scale_ranges, self.num_grids, feature_size_list): 
            mask_gt_batch = []
            cate_gt_batch = []
            index_gt_batch = []

            # index_gt_batch_xy = []
            ### 在batch中遍历
            for area, mask, cate, center_h, center_w, ymin, xmin, ymax, xmax  \
                in zip(label['area'], label["mask"], label["category"], label["ycenter"], label["xcenter"], \
                    label['ymin'], label['xmin'], label['ymax'], label['xmax']):
                ### 为每种grid初始化ground truth
                mask_label = np.zeros([1, feature_size[0], feature_size[1], num_grid**2], dtype = np.int64) ## 实例gt,最原始的, [h,w, s^2]
                cate_label = np.zeros([1, num_grid, num_grid, self.num_classes], dtype = np.int64)                            ## grid格子，值为类别，[s,s]
                index_label = np.zeros([1, num_grid**2], dtype = np.bool)                                ## s*s格子，bool型，表示有没有实例  
                
                hit_indices = (area >= lower_bound) & (area <= upper_bound)
                # print("hit shape", hit_indices.shape)
                num_hit_indices = tf.reduce_sum(tf.cast(hit_indices, tf.int64)).numpy()
                if num_hit_indices == 0:
                    cate_gt_batch.append(cate_label)
                    mask_gt_batch.append(mask_label)
                    index_gt_batch.append(index_label)
                    continue
                
                # print("mask shape", mask.shape)
                mask_hit = tf.boolean_mask(mask, hit_indices, axis= 0)
                cate_hit = tf.boolean_mask(cate, hit_indices, axis=0)
                center_h_hit = tf.boolean_mask(center_h, hit_indices, axis=0)
                center_w_hit = tf.boolean_mask(center_w, hit_indices, axis=0)
                ymin_hit = tf.boolean_mask(ymin, hit_indices, axis=0)
                xmin_hit = tf.boolean_mask(xmin, hit_indices, axis=0)
                ymax_hit = tf.boolean_mask(ymax, hit_indices, axis=0)
                xmax_hit = tf.boolean_mask(xmax, hit_indices, axis=0)

                half_hs = [0.5 * (_ymax - _ymin) * self.sigma for _ymax, _ymin in zip(ymax_hit, ymin_hit)]
                half_ws = [0.5 * (_xmax - _xmin) * self.sigma for _xmax, _xmin in zip(xmax_hit, xmin_hit)]
                
                # print("num:", num_instance)
                for i in range(num_hit_indices):
                    # print("cate: ", cate[i])
                    ### 中心坐标归一化后在网格中真实坐标
                    coord_h = int( center_h_hit[i] * num_grid )
                    coord_w = int( center_w_hit[i] * num_grid )

                    ### 论文3.3.1引入的gt box
                    top_box = max(0, int( (center_h_hit[i] - half_hs[i]) * num_grid) )
                    down_box = min(num_grid - 1, int( (center_h_hit[i] + half_hs[i]) * num_grid ) )
                    left_box = max(0, int( (center_w_hit[i] - half_ws[i]) * num_grid ) )
                    right_box  = min(num_grid - 1, int( (center_w_hit[i] + half_ws[i]) * num_grid ) )

                    ###与中心点3x3区域取交
                    top = max(top_box, coord_h - 1)
                    down = min(down_box, coord_h + 1)
                    left = max(left_box, coord_w - 1)
                    right = min(right_box, coord_w + 1)

                    # if np.sum(cate_label[0, top : down + 1, left : right + 1]) > 0:
                    #     print("占用", np.sum(cate_label[0, top : down + 1, left : right + 1] > 0))
                    cate_label[0, top : down + 1, left : right + 1,cate_hit[i] - 1] = 1

                    ### 将gt_mask resize到实际大小，与feature map 大小一致
                    # mask_resize = cv2.resize(mask, feature_size , interpolation=cv2.INTER_NEAREST)
                    mask_resize = tf.image.resize(tf.expand_dims(mask_hit[i],-1), feature_size, method='bilinear')
                    mask_resize = tf.cast(mask_resize + 0.5, tf.int64)
                    for i in range(top, down + 1):
                        for j in range(left, right + 1):
                            index = int(i * num_grid + j)
                            mask_label[0, :, :, index] = mask_resize[:,:,0]
                            index_label[0, index] = True
                # tmp, _ = tf.unique(tf.reshape(cate_label,[-1]))
                # print(tmp)
                cate_gt_batch.append(cate_label)
                mask_gt_batch.append(mask_label)
                index_gt_batch.append(index_label)
                # index_gt_batch_xy.append(tf.where(cate_label > 0))
            cate_gts.append(tf.concat(cate_gt_batch,axis = 0))
            mask_gts.append(tf.concat(mask_gt_batch,axis = 0))
            index_gts.append(tf.concat(index_gt_batch,axis = 0))
        return mask_gts, cate_gts, index_gts

    def calc_loss(self, preds, label):

        ins_preds_y, ins_preds_x, cate_preds = preds

        feature_size_list = [ featmap.shape.as_list()[1:3] for featmap in ins_preds_y]
        mask_gts, cate_gts, index_gts = self.ground_truth(label,feature_size_list)

        if 0:
            for id in range(len(cate_preds)):
                cate_pred = cate_preds[id]
                cate_gt = cate_gts[id]
                print(
                    "cate pred max: {}, min: {} ".format( 
                        tf.reduce_sum(tf.cast(cate_pred > 0, tf.float32)).numpy(), 
                        # tf.reduce_sum(tf.cast(cate_pred <= 0, tf.float32)).numpy()
                        tf.reduce_max(cate_pred)
                        ) 
                    )
                print(
                    "cate gt max: {}, min: {} ".format( 
                        tf.reduce_sum(tf.cast(cate_gt > 0, tf.float32)).numpy(), 
                        # tf.reduce_sum(tf.cast(cate_gt <= 0, tf.float32)).numpy()
                        tf.reduce_max(cate_gt)
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
        # ins_labels_pred = []
        # for mask_batch, index_batch in zip(ins_preds, index_gts):
        #     ins_labels = []
        #     for mask_label, index_label in zip( mask_batch, index_batch ):
        #         # mask_label_transpose = tf.transpose(mask_label,(2,0,1))
        #         # ins_label = tf.transpose(tf.boolean_mask(mask_label_transpose, index_label),(1,2,0))
        #         ins_label = tf.boolean_mask(mask_label, index_label, axis = 2)
        #         ins_labels.append(ins_label)
        #     ins_labels_pred.append(tf.concat(ins_labels,axis = -1))

        ### pred instance y, x
        ins_labels_pred_y = []
        ins_labels_pred_x = []
        for mask_batch_y, mask_batch_x, cate_batch  in zip(ins_preds_y, ins_preds_x, cate_gts):
            ins_labels_y = []
            shape = [mask_batch_y.shape[1],mask_batch_y.shape[2],0]
            ins_labels_y.append(tf.zeros(shape, dtype = tf.float32))
            ins_labels_x = []
            ins_labels_x.append(tf.zeros(shape, dtype = tf.float32))
            for mask_label_y, mask_label_x, cate_label in zip( mask_batch_y, mask_batch_x, cate_batch ):
                # print("y shape: {}, x shape: {}, cate shape: {}".format(mask_label_y.shape, mask_label_x.shape, cate_label.shape))
                cate_label = tf.reduce_max(cate_label,axis=-1)
                coordinate = tf.where(cate_label > 0)
                # print("coord shape: ",coordinate.shape)
                if not coordinate.shape[0] == 0:
                    # print("mask_label_y: ", mask_label_y.shape)
                    # ins_label_y = tf.boolean_mask(mask_label_y, coordinate[:,0], axis = 2)
                    # ins_label_y = gather_axis(mask_label_y, coordinate[:,0], axis = -1)
                    ins_label_y = tf.gather(mask_label_y, coordinate[:,0], axis = -1)
                    ins_labels_y.append(ins_label_y)
                    # ins_label_x = tf.boolean_mask(mask_label_x, coordinate[:,1], axis = 2)
                    # ins_label_x = gather_axis(mask_label_x, coordinate[:,1], axis = -1)
                    ins_label_x = tf.gather(mask_label_x, coordinate[:,1], axis = -1)
                    ins_labels_x.append(ins_label_x)
            ins_labels_pred_y.append(tf.concat(ins_labels_y, axis = -1))
            ins_labels_pred_x.append(tf.concat(ins_labels_x, axis = -1))

        # ins_index_gt = []
        # for index_batch in zip(index_gts):
        #     index_labels = []
        #     for ins_ind in zip(index_batch):
        #         index_labels.append(tf.reshape(ins_ind, [-1]))
        #     ins_index_gt.append(tf.concat(index_labels,axis = 0))
        
        # flatten_ins_index_gt = tf.concat(ins_index_gt, axis = 0)

        # num_instance = tf.reduce_sum(tf.cast(flatten_ins_index_gt, tf.int64))

        ### dice loss
        loss_ins = []
        num_ins = 0.
        for ins_pred_y, ins_pred_x, ins_gt  in zip(ins_labels_pred_y, ins_labels_pred_x, ins_labels_gt):
            mask_n = ins_pred_y.shape[2]
            if mask_n == 0:
                continue
            num_ins += mask_n
            pred = tf.math.sigmoid(ins_pred_y) * tf.math.sigmoid(ins_pred_x)
            # print("pred shape: ", pred.shape)
            # print("index gt shape:", ins_gt.shape)
            loss_ins.append(dice_loss(pred,ins_gt))
        loss_ins = tf.reduce_mean(tf.concat(loss_ins,0)) * 3.0
        # loss_ins = tf.reduce_sum(tf.concat(loss_ins,0)) * 3.0 / (num_ins + 1e-9)

        ### cate
        cate_gt_list = []
        for cate_gt_batch in cate_gts:   ### cate_batch's shape maybe  [batch_size,40,40,num_classes]
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
        # loss_cate = focal_loss(flatten_cate_preds, flatten_cate_gt, num_ins + 1) * 1.0
        loss_cate = sigmoid_focal_loss(flatten_cate_preds, flatten_cate_gt, avg_factor=num_ins+1)
        # loss_cate = se_loss(flatten_cate_preds, flatten_cate_gt, avg_factor=num_ins+1)

        return dict(loss_cate=loss_cate,loss_ins=loss_ins)

    def get_seg(self, all_preds, img_metas, cfg, rescale = None):

        seg_preds_y, seg_preds_x, cate_preds = all_preds
        assert len(seg_preds_y) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds_y[0].shape[1:3]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                tf.reshape(cate_preds[i][img_id],[-1, self.num_classes]) for i in range(num_levels)
            ]
            seg_pred_list_x = [
                seg_preds_x[i][img_id] for i in range(num_levels)
            ]
            seg_pred_list_y = [
                seg_preds_y[i][img_id] for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = tf.concat(cate_pred_list, axis = 0)
            seg_pred_list_x = tf.concat(seg_pred_list_x, axis = -1)
            seg_pred_list_y = tf.concat(seg_pred_list_y, axis = -1)
            if DEBUG:
                print("cate_pred_list shape: {} ".format(cate_pred_list.shape))
                print("seg_pred_list_x shape: {} ".format(seg_pred_list_x.shape))
                print("seg_pred_list_y shape: {} ".format(seg_pred_list_y.shape))
            result = self.get_seg_single(cate_pred_list, seg_pred_list_y, seg_pred_list_x,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list


    def get_seg_single(self,
                        cate_preds,
                        seg_preds_y,
                        seg_preds_x,
                        featmap_size,
                        img_shape,
                        ori_shape,
                        scale_shape,
                        cfg,
                        rescale=False):
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # trans_size = tf.cumsum(tf.pow(self.num_grids,2))
        num_grids_square = [grid**2 for grid in self.num_grids]
        trans_size = np.cumsum(num_grids_square)
        # trans_diff = tf.ones(trans_size[-1], dtype = tf.int64)
        # num_grids = tf.ones(trans_size[-1], dtype = tf.int64)
        seg_size = np.cumsum(self.num_grids)
        # seg_diff = tf.ones(trans_size[-1], dtype = tf.int64)
        # strides = tf.ones(trans_size[-1], dtype = tf.int64)

        n_stage = len(self.num_grids)
        trans_diff_list = []
        seg_diff_list = []
        num_grids_list = []
        strides_list = []

        trans_diff_list.append(tf.zeros(trans_size[0], dtype = tf.int64))
        seg_diff_list.append(tf.zeros(trans_size[0], dtype = tf.int64))
        num_grids_list.append(tf.ones(trans_size[0], dtype = tf.int64) * self.num_grids[0])
        strides_list.append(tf.ones(trans_size[0], dtype = tf.int64) * self.strides[0])
        # trans_diff[:trans_size[0]] *= 0
        # seg_diff[:trans_size[0]] *= 0
        # num_grids[:trans_size[0]] *= self.num_grids[0]
        # strides[:trans_size[0]] *= self.strides[0]

        for ind_ in range(1, n_stage):
            # trans_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= trans_size[ind_ - 1]
            # seg_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= seg_size[ind_ - 1]
            # num_grids[trans_size[ind_ - 1]:trans_size[ind_]] *= self.num_grids[ind_]
            # strides[trans_size[ind_ - 1]:trans_size[ind_]] *= self.strides[ind_]
            trans_diff_list.append(tf.ones(num_grids_square[ind_], dtype = tf.int64) * trans_size[ind_ - 1])
            seg_diff_list.append(tf.ones(num_grids_square[ind_], dtype = tf.int64) * seg_size[ind_ - 1])
            num_grids_list.append(tf.ones(num_grids_square[ind_], dtype = tf.int64) * self.num_grids[ind_])
            strides_list.append(tf.ones(num_grids_square[ind_], dtype = tf.int64) * self.strides[ind_])

        # trans_diff *= tf.concat(trans_diff_list, axis = 0)
        # seg_diff *= tf.concat(seg_diff_list, axis = 0)
        # num_grids *= tf.concat(num_grids_list, axis = 0)
        # strides *= tf.concat(strides_list, axis = 0)
        trans_diff = tf.concat(trans_diff_list, axis = 0)
        seg_diff = tf.concat(seg_diff_list, axis = 0)
        num_grids = tf.concat(num_grids_list, axis = 0)
        strides = tf.concat(strides_list, axis = 0)

        # process.
        inds = tf.where(cate_preds > cfg["score_thr"])
        cate_scores = tf.gather_nd(cate_preds, inds)
        trans_diff = tf.gather(trans_diff, inds[:, 0], axis=0)
        seg_diff = tf.gather(seg_diff, inds[:, 0], axis=0)
        num_grids = tf.gather(num_grids, inds[:, 0], axis=0)
        strides = tf.gather(strides, inds[:, 0], axis=0)
        # print("------------------shape--------------------")
        # print("inds shape: ", inds.shape)# print("cate_scores shape: ", cate_scores.shape)
        # print("trans_diff shape: ", trans_diff.shape)
        # print("seg_diff shape: ", seg_diff.shape)
        # print("num_grids shape: ", num_grids.shape)
        # print("strides shape: ", strides.shape)
        # print("------------------shape--------------------")
        y_inds = (inds[:, 0] - trans_diff) // num_grids
        x_inds = (inds[:, 0] - trans_diff) % num_grids
        y_inds += seg_diff
        x_inds += seg_diff
        
        cate_labels = inds[:, 1]
        if DEBUG:
            print("seg_preds_x shape: ", seg_preds_x.shape)
            print("x_inds shape:", x_inds.shape)
            print("y_inds shape:", y_inds.shape)
        seg_masks_soft = tf.gather(seg_preds_x, x_inds, axis=-1) * tf.gather(seg_preds_y, y_inds, axis=-1)
        if DEBUG:
            print("seg_masks_soft shape", seg_masks_soft.shape)
        seg_masks = seg_masks_soft > cfg["mask_thr"]
        # sum_masks = seg_masks.sum((1, 2)).float()
        sum_masks = tf.reduce_sum(tf.cast(seg_masks_soft > cfg["mask_thr"],tf.float32),axis=[0,1])
        if DEBUG:
            print("sum_masks shape: ", sum_masks.shape)
            print("strides shape: ", strides.shape)
        # keep = sum_masks > strides  ### 一维向量
        keep = tf.squeeze(tf.where(sum_masks > tf.cast(strides,tf.float32)), axis = -1)

        seg_masks_soft = tf.gather(seg_masks_soft, keep, axis= -1)
        seg_masks = tf.gather(seg_masks, keep, axis= -1)
        cate_scores = tf.gather(cate_scores, keep, axis= -1)
        sum_masks = tf.gather(sum_masks, keep, axis= -1)
        cate_labels = tf.gather(cate_labels, keep, axis= -1)

        if DEBUG:
            print("keep shape: ", keep.shape)
            print("seg_masks_soft shape: ", seg_masks_soft.shape)
        # mask scoring
        seg_score = tf.reduce_sum(seg_masks_soft * tf.cast(seg_masks, tf.float32),axis=[0,1]) / sum_masks
        cate_scores *= seg_score

        if len(cate_scores) == 0:
            return None

        ###----------------------------------------------------------------------------###
        # sort and keep top nms_pre
        if DEBUG:
            print("cate_scores shape: ", cate_scores.shape)
        sort_inds = tf.argsort(cate_scores, direction='DESCENDING')
        if DEBUG:
            print("sort_inds shape1: ", sort_inds.shape)
        if len(sort_inds) > cfg["nms_pre"]:
            sort_inds = sort_inds[:cfg["nms_pre"]]
        if DEBUG:
            print("sort_inds shape2: ", sort_inds.shape)
            print("seg_masks_soft shape1: ", seg_masks_soft.shape)
        seg_masks_soft = tf.gather(seg_masks_soft, sort_inds, axis=-1)
        if DEBUG:
            print("seg_masks_soft shape2: ", seg_masks_soft.shape)
        seg_masks = tf.gather(seg_masks, sort_inds, axis=-1)
        cate_scores = tf.gather(cate_scores, sort_inds, axis=-1)
        sum_masks = tf.gather(sum_masks, sort_inds, axis=-1)
        cate_labels = tf.gather(cate_labels, sort_inds, axis=-1)

        if DEBUG:
            print("seg_masks shape: ", seg_masks.shape)
            print("cate_scores shape: ", cate_scores.shape)
            print("sum_masks shape: ", sum_masks.shape)
            print("cate_labels shape: ", cate_labels.shape)
        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg["kernel"], sigma=cfg["sigma"], sum_masks=sum_masks)

        if DEBUG:
            print("cate_scores shape:", cate_scores.shape)
        keep = tf.squeeze(tf.where(cate_scores >= cfg["update_thr"]), axis = -1)
        seg_masks_soft = tf.gather(seg_masks_soft, keep, axis = -1)
        cate_scores = tf.gather(cate_scores, keep, axis=-1)
        cate_labels = tf.gather(cate_labels, keep, axis=-1)

        if DEBUG:
            print("keep shape:", keep.shape)
            print("seg_masks_soft shape:", seg_masks_soft.shape)
            print("cate_scores shape:", cate_scores.shape)
            print("cate_labels shape:", cate_labels.shape)

        # sort and keep top_k
        sort_inds = tf.argsort(cate_scores, direction='DESCENDING')
        if len(sort_inds) > cfg["max_per_img"]:
            sort_inds = sort_inds[:cfg["max_per_img"]]
        seg_masks_soft = tf.gather(seg_masks_soft, sort_inds, axis = -1)
        cate_scores = tf.gather(cate_scores, sort_inds, axis = -1)
        cate_labels = tf.gather(cate_labels, sort_inds, axis = -1)

        # seg_masks_soft = F.interpolate(seg_masks_soft.unsqueeze(0),
        #                             size=upsampled_size_out,
        #                             mode='bilinear')[:, :, :h, :w]
        # print("***************\nseg_masks_soft shape: ", seg_masks_soft.shape)
        seg_masks_soft = tf.image.resize(seg_masks_soft, size = upsampled_size_out, method='bilinear')
        # seg_masks = F.interpolate(seg_masks_soft[:h, :w, ...],
        #                        size=ori_shape[:2],
        #                        mode='bilinear').squeeze(0)
        seg_masks = tf.image.resize(seg_masks_soft[:h, :w, ...], ori_shape[:2], method='bilinear')
        seg_masks = seg_masks > cfg["mask_thr"]
        return seg_masks, cate_labels, cate_scores
