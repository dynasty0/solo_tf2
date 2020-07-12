import tensorflow as tf
from backbone import *
from neck import *
from head import *

class SOLO(tf.keras.Model):
    def __init__(self, cfg):
        super(SOLO,self).__init__()
        self.backbone = eval(cfg['backbone']['type'])(cfg['backbone'])
        # self.backbone = resnet101(pretrained=False, data_format="channels_last")
        self.neck = eval(cfg['neck']['type'])(cfg['neck'])
        self.head = eval(cfg['head']['type'])(cfg['head'])

    def call(self, input, eval = False, training=True):
        features = self.backbone(input, training=training)
        features = self.neck(features, training=training)
        features = self.head(features, eval=eval, training=training)

        self.all_preds = features
        return features

    def calc_loss(self, preds, gt_label):
        loss = self.head.calc_loss(preds, gt_label)
        total_loss = loss["loss_cate"] + loss["loss_ins"]
        return loss["loss_ins"], loss["loss_cate"], total_loss

    def get_seg(self, img_metas, cfg, rescale = None):
        return self.head.get_seg(self.all_preds, img_metas, cfg, rescale = None)