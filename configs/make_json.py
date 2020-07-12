import json

model = dict(
    type='SOLO',
    backbone=dict(
        type="ResNet101",
        input_shape=[1200,1200,3]
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    head=dict(
        type='DecoupledHead',
        num_classes=34,
        in_channels=256,
        stacked_convs=7,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        with_deform=False,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
    ))

dataset=dict(
    type="FutureDataset",
    data_path = "/home/notebook/data/personal/3d-future/tfrecords",
    data_shape = 1200,
    batch_size = 2,
    epoch_train = 12,
    num_train = 10000,
    num_val = 2144,
)

train=dict(
    base_learning_rate = 0.001,
    decay_epoches=[8,11],
    loss_step = 100,
    save_step = 1000,
    test_step = 10000,
    ckpt_path="./saved_models/future/checkpoints",
    best_model_path = "./saved_models/future/best_model",
    saved_path = "./saved_models/future/model_0",
)

test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)

### 写成json
model_config = dict()
model_config['model'] = model
model_config['dataset'] = dataset
model_config['train'] = train
model_config['test_cfg'] = test_cfg
with open('future3d.json','w') as f:
    json.dump(model_config,f,indent=4)


# ### 读、解析json
# with open("model.json",'r') as f:
#     cfg = json.load(f)
# model = cfg['model']
# print(model['backbone'])

# from util import parse_json 
# print(parse_json("model.json"))