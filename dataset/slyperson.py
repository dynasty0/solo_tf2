import tensorflow as tf
from .augmentation import *


# Create a dictionary describing the features.
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'filename': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'num': tf.io.FixedLenFeature([], tf.int64),
    'mask': tf.io.VarLenFeature(tf.string),
    'category': tf.io.VarLenFeature(tf.int64),
    'area': tf.io.VarLenFeature(tf.float32),
    'ymin': tf.io.VarLenFeature(tf.float32),
    'xmin': tf.io.VarLenFeature(tf.float32),
    'ymax': tf.io.VarLenFeature(tf.float32),
    'xmax': tf.io.VarLenFeature(tf.float32),
    'ycenter': tf.io.VarLenFeature(tf.float32),
    'xcenter': tf.io.VarLenFeature(tf.float32),
}

class PersonDataset():
    def __init__(self, path, augmentation = True, shape = (512,512), max_size = 20):
        self.path = path
        self.dataset = tf.data.TFRecordDataset(self.path)
        self.max_size = max_size
        self.shape = shape
        self.augmentation = augmentation
        # self.output_dict = dict()

    def _parse_image_function(self,example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        parsed_tensors = tf.io.parse_single_example(example_proto, image_feature_description)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=""
                    )
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0)
        return parsed_tensors

    def _decode_image(self, parsed_tensors):
        image = tf.io.decode_jpeg(parsed_tensors['image'])
        image.set_shape([None, None, 3])
        image = tf.cast(image, tf.float32)
        return image
    
    def _decode_masks(self, parsed_tensors):
        def _decode_png_mask(png_bytes):
            mask = tf.squeeze(
                tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
            mask = tf.cast(mask, dtype=tf.float32)
            mask.set_shape([None, None])
            return mask

        height = parsed_tensors['height']
        width = parsed_tensors['width']
        masks = parsed_tensors['mask']
        return tf.cond(
            pred=tf.greater(tf.size(input=masks), 0),
            true_fn=lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
            false_fn=lambda: tf.zeros([0, height, width], dtype=tf.float32)
            )

    def decode(self,data):
        # data['image'] = self.decode_img(data['image'])
        # data['image'] = tf.reshape(data['image'],(data['height'],data['width'],3))
        # data['image'] = tf.cast(data['image'],tf.float32) *2/255 - 1
        image = self._decode_image(data)
        mask = self._decode_masks(data)  ## [n_mask,h,w]
        # mask = tf.transpose(mask, perm = [1,2,0]) ## [h,w,n_mask]
        ymin = data["ymin"]
        xmin = data["xmin"]
        ymax = data["ymax"]
        xmax = data["xmax"]
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis = 1)
        classes = data["category"]

        # uniques = np.unique(image.numpy())
        # if len(uniques) < 10 :
        #     print("unique: {}".format(uniques) )
        
        if self.augmentation:
            mask = tf.expand_dims(mask, axis = -1)
            image, bboxes, mask, classes = random_augmentation(image, bboxes, mask, classes, self.shape[0], self.shape[0])
        if tf.shape(classes)[0] == 1 and len( tf.shape(mask) ) < 3:
            mask = tf.expand_dims(mask, axis=0)
        # pad to same size for batch
        pad_size = self.max_size - tf.shape(classes)[0]
        paded_xmin = tf.zeros([pad_size])
        xmin = tf.concat([bboxes[:,1],paded_xmin], axis = 0)
        paded_xmax = tf.zeros([pad_size])
        xmax = tf.concat([bboxes[:,3],paded_xmax], axis = 0)
        paded_ymin = tf.zeros([pad_size])
        ymin = tf.concat([bboxes[:,0],paded_ymin], axis = 0)
        paded_ymax = tf.zeros([pad_size])
        ymax = tf.concat([bboxes[:,2],paded_ymax], axis = 0)

        paded_ycenter = tf.zeros([pad_size])
        ycenter = tf.concat([ (bboxes[:,0] + bboxes[:,2]) / 2, paded_ycenter], axis = 0)
        paded_xcenter= tf.zeros([pad_size])
        xcenter = tf.concat([ (bboxes[:,1] + bboxes[:,3]) / 2, paded_xcenter], axis = 0)
        paded_area = tf.zeros([pad_size])
        # _area = tf.reduce_sum(mask, axis = [0,1])
        _area = tf.math.sqrt( (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1]) ) * self.shape[0]
        area = tf.concat([_area, paded_area], axis = 0)

        paded_category = tf.zeros([pad_size],dtype=tf.int64)
        category = tf.concat([classes,paded_category], axis = 0)

        
        # paded_mask = tf.zeros([self.shape[0], self.shape[1], pad_size])
        # mask = tf.concat([mask, paded_mask], axis = -1)
        paded_mask = tf.zeros([pad_size, self.shape[0], self.shape[1]])
        mask = tf.concat([mask, paded_mask], axis = 0)


        image = image * 2. / 255. - 1.
        label = {
            'num': data['num'],
            'mask': mask,
            'ymin': ymin,
            'xmin': xmin,
            'ymax': ymax,
            'xmax': xmax,
            'ycenter': ycenter,
            'xcenter': xcenter,
            'area': area,
            'category': category,
        }
        return image, label

    def parse(self,data):
        data = self._parse_image_function(data)
        return self.decode(data)

    def run(self, epoches, batch_size):
        self.dataset = self.dataset.shuffle(buffer_size=2048)
        # self.dataset = self.dataset.map(self._parse_image_function)   ### for test below
        # self.dataset = self.dataset.map(self._decode_image)
        self.dataset = self.dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # return self.dataset
        # self.dataset = self.dataset.shuffle(1000).repeat(epoches)
        
        
        self.dataset = self.dataset.repeat(epoches)
        
        self.dataset = self.dataset.batch(batch_size)
        return self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  
