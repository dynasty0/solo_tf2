import tensorflow as tf
from configs.util import parse_json
from model import SOLO
from dataset import *
import sys
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

json_path = "./configs/future3d.json"
cfg = parse_json(json_path)
model = SOLO(cfg['model'])

##### metrics define
train_cate_loss = tf.keras.metrics.Mean(name='train_cate_loss')
train_ins_loss = tf.keras.metrics.Mean(name='train_ins_loss')

test_cate_loss = tf.keras.metrics.Mean(name='test_cate_loss')
test_ins_loss = tf.keras.metrics.Mean(name='test_ins_loss')
test_total_loss = tf.keras.metrics.Mean(name='test_total_loss')

# @tf.function
def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss_ins, loss_cate, total_loss = model.calc_loss(predictions,labels)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # var_list = []
    # for var in model.trainable_variables:
    #     if "mask_branch" in var.name:
    #         var_list.append(var) 
    # gradients = tape.gradient(total_loss, var_list)
    # optimizer.apply_gradients(zip(gradients, var_list))
    train_cate_loss.update_state(loss_cate)
    train_ins_loss.update_state(loss_ins)
    return loss_ins, loss_cate


def test_step(model, images, labels):
    predictions = model(images, training=False)
    loss_ins, loss_cate, total_loss = model.calc_loss(predictions,labels)
    test_cate_loss.update_state(loss_cate)
    test_ins_loss.update_state(loss_ins)
    test_total_loss.update_state(total_loss)
    return loss_ins, loss_cate, total_loss


cfg_dataset = cfg['dataset']
cfg_train = cfg['train']

batch_size = cfg_dataset['batch_size']
epoches = cfg_dataset['epoch_train']

num_train = cfg_dataset['num_train']
num_test = cfg_dataset['num_val']

lr_base = cfg_train['base_learning_rate']
decay_epoches = cfg_train['decay_epoches']

print("init train & val dataset")

train_record_path = os.path.join(cfg_dataset['data_path'], "train.tfrecords") 
train_dataset = eval(cfg_dataset['type'])(train_record_path, shape = (cfg_dataset['data_shape'],cfg_dataset['data_shape']))
train_dataset = train_dataset.run(epoches=epoches, batch_size=batch_size)

test_record_path = os.path.join(cfg_dataset['data_path'], "val.tfrecords") 
test_dataset = eval(cfg_dataset['type'])(test_record_path, shape = (cfg_dataset['data_shape'],cfg_dataset['data_shape']))
test_dataset = test_dataset.run(epoches=1, batch_size=1)
print("init dataset end")

##### calc learning rate function
boundaries = []
values = []

values.append(lr_base)
lr = lr_base
count = 1
for epoch in decay_epoches:
    lr = lr / 2 if count % 2 else lr / 5
    count += 1
    step = num_train//batch_size * epoch
    boundaries.append(step)
    values.append(lr)

learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9,decay = 1e-4)


# setup checkpoints manager
# checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model, best_model_loss=tf.Variable(1.0e9))
manager = tf.train.CheckpointManager(
    checkpoint, directory=cfg_train['ckpt_path'], max_to_keep=5
)
# restore from latest checkpoint and iteration
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}, global step: {}".format(manager.latest_checkpoint, int(checkpoint.step)))
else:
    print("Initializing from scratch.")


step_to_test = cfg_train['test_step']
step_to_save = cfg_train["save_step"]
step_to_eval_loss = cfg_train["loss_step"]

step_epoch = num_train//batch_size
# step = 0
# best_model_loss = 1e9
# checkpoint.best_model_loss.assign(1e9)

print("start training")
for image, label in train_dataset:

    step = int(checkpoint.step)
    checkpoint.step.assign_add(1)

    loss_ins, loss_cate = train_step(model, image, label,optimizer)
    # print("loss: ", loss_cate.numpy(), loss_ins.numpy())
    ##### processbar
    _num = step % step_to_eval_loss
    if _num == 1:
        print('train next {} steps, batch = {} : '.format(step_to_eval_loss, batch_size))
    string = ''
    string = string + '\b' * len(str(_num))
    if _num == 0:
        string = string + '\b#100\n'
    else:
        string = string + '#' + str(_num)
    print(string, end='')
    sys.stdout.flush()
    ##### end of processbar

    ##### eval training loss every step_to_eval_loss(100) steps        
    if step % step_to_eval_loss == 0:
        # string = ''
        print("epoch {}, step {}, cate loss: {} , ins loss: {} ".format( \
            batch_size* step //num_train, step % (num_train // batch_size) , train_cate_loss.result(), train_ins_loss.result()) 
        )
        train_cate_loss.reset_states()
        train_ins_loss.reset_states()

    ##### test on val set
    if step % step_to_test == 0 or step % step_epoch == 0:
        if step % step_to_test == 0 and not step % step_epoch == 0:
            print("{} steps end, now test on val set, take 100 random images to test".format(step_to_test))
            for _image, _label in test_dataset.take(100):
                loss_ins, loss_cate, total_loss = test_step(model, _image, _label)
                # print("loss: ", loss_cate.numpy(), loss_ins.numpy())
            print("epoch {}, cate loss: {} , ins loss: {} ".format( \
                batch_size * step //num_train, test_cate_loss.result(), test_ins_loss.result()) 
            )
        elif step % step_epoch == 0:
            print("{} epoch end, now test on val set. ".format(step // step_epoch))
            for _image, _label in test_dataset:
                loss_ins, loss_cate, total_loss = test_step(model, _image, _label)
                # print("loss: ", loss_cate.numpy(), loss_ins.numpy())
            print("epoch {}, cate loss: {} , ins loss: {} ".format( \
                batch_size * step //num_train, test_cate_loss.result(), test_ins_loss.result()) 
            )
        
        

        # def representative_data_gen():
        #     for image, label in test_dataset.take(100):
        #         yield [image]

        if test_total_loss.result() < checkpoint.best_model_loss.numpy():
            print("found a better model")
            checkpoint.best_model_loss.assign(test_total_loss.result())
            save_path = cfg_train['best_model_path']
            model.save(save_path, save_format='tf')
            print("original model saved at {}".format(save_path))
            print("convert model to tflite...")
            converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # converter.target_ops = [tf.uint8]
            # converter.representative_dataset = representative_data_gen
            # # converter.inference_input_type = tf.uint8 
            # # converter.inference_output_type = tf.uint8
            # converter.inference_type = tf.uint8
            tflite_model = converter.convert()
            open(save_path + '/model.tflite',"wb").write(tflite_model)
            print("tflite model saved at {}".format(save_path + '/model.tflite'))

        test_cate_loss.reset_states()
        test_ins_loss.reset_states()
        test_total_loss.reset_states()

    ##### save ckpt and tf model
    if step % step_to_save == 0:
        ckpt_path = manager.save()
        print("Saved checkpoint for step {}: {}".format( step, ckpt_path))
        save_path = cfg_train["saved_path"]
        model.save(save_path, save_format='tf')
        print("save model at step {}, model saved at {} .".format( step, save_path))
