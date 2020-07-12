import tensorflow as tf

def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (h, w, n)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        # sum_masks = seg_masks.sum((1, 2)).float()
        sum_masks = tf.reduce_sum(tf.cast(seg_masks, tf.float32), axis=[0,1])
    # seg_masks = seg_masks.reshape(n_samples, -1).float()
    seg_masks = tf.reshape(tf.cast(seg_masks, tf.float32),[-1, n_samples])
    # inter.
    # inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    inter_matrix = tf.matmul(tf.transpose(seg_masks), seg_masks)
    # union.
    # sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # print("******************\nsum_masks shape: ", sum_masks.shape)
    # print("n_samples: ", n_samples)
    sum_masks_x = tf.tile(tf.expand_dims(sum_masks,0),[n_samples,1])
    # print("sum_masks_x shape: ", sum_masks_x.shape)
    # iou.
    _iou_matrix = inter_matrix / (sum_masks_x + tf.transpose(sum_masks_x) - inter_matrix)
    iou_matrix = tf.linalg.band_part(_iou_matrix, 0, -1) - tf.linalg.band_part(_iou_matrix, 0, 0)
    # label_specific matrix.
    # cate_labels_x = cate_labels.expand(n_samples, n_samples)
    cate_labels_x = tf.tile(tf.expand_dims(cate_labels, 0), [n_samples, 1] )
    _label_matrix = tf.cast(cate_labels_x == tf.transpose(cate_labels_x), tf.float32)
    label_matrix = tf.linalg.band_part(_label_matrix, 0, -1) - tf.linalg.band_part(_label_matrix, 0, 0)
    # IoU compensation
    # compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = tf.reduce_max(iou_matrix * label_matrix, axis=0)
    # compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)
    compensate_iou = tf.transpose(tf.tile(tf.expand_dims(compensate_iou, 0), [n_samples, 1]))

    # IoU decay 
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        # decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        decay_matrix = tf.math.exp(-1.0 * sigma * (decay_iou ** 2))
        # compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        compensate_matrix = tf.math.exp(-1.0 * sigma * (compensate_iou ** 2))
        # decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
        decay_coefficient = tf.reduce_min(decay_matrix / compensate_matrix, axis=0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        # decay_coefficient, _ = decay_matrix.min(0)
        decay_coefficient = tf.reduce_min(decay_matrix, axis=0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update