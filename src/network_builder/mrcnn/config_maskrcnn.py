import numpy as np

#
# Base Configuration Class 
class ConfigMaskrcnn(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    def __init__(self, json_config):
        super(ConfigMaskrcnn, self).__init__()
        self.GPU_COUNT = json_config['GPU_COUNT']
        self.IMAGES_PER_GPU = json_config['IMAGES_PER_GPU']
        self.NUM_CLASSES = json_config['NUM_CLASSES']

        self.STEPS_PER_EPOCH = json_config['STEPS_PER_EPOCH']
        self.VALIDATION_STEPS = json_config['VALIDATION_STEPS']
        self.BACKBONE = json_config['BACKBONE']
        self.COMPUTE_BACKBONE_SHAPE = json_config['COMPUTE_BACKBONE_SHAPE']
        self.BACKBONE_STRIDES = json_config['BACKBONE_STRIDES']
        self.FPN_CLASSIF_FC_LAYERS_SIZE =json_config['FPN_CLASSIF_FC_LAYERS_SIZE']
        self.TOP_DOWN_PYRAMID_SIZE = json_config['TOP_DOWN_PYRAMID_SIZE']
        self.RPN_ANCHOR_SCALES = (
            json_config['RPN_ANCHOR_SCALES'][0],
            json_config['RPN_ANCHOR_SCALES'][1],
            json_config['RPN_ANCHOR_SCALES'][2],
            json_config['RPN_ANCHOR_SCALES'][3],
            json_config['RPN_ANCHOR_SCALES'][4])
        self.RPN_ANCHOR_RATIOS = json_config['RPN_ANCHOR_RATIOS']
        self.RPN_ANCHOR_STRIDE = json_config['RPN_ANCHOR_STRIDE']
        self.RPN_NMS_THRESHOLD = json_config['RPN_NMS_THRESHOLD']
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = json_config['RPN_TRAIN_ANCHORS_PER_IMAGE']
        self.PRE_NMS_LIMIT = json_config['PRE_NMS_LIMIT']
        self.POST_NMS_ROIS_TRAINING = json_config['POST_NMS_ROIS_TRAINING']
        self.POST_NMS_ROIS_INFERENCE = json_config['POST_NMS_ROIS_INFERENCE']
        self.USE_MINI_MASK = json_config['USE_MINI_MASK']
        self.MINI_MASK_SHAPE = (json_config['MINI_MASK_SHAPE'][0],json_config['MINI_MASK_SHAPE'][1])
        self.IMAGE_RESIZE_MODE = json_config['IMAGE_RESIZE_MODE']
        self.IMAGE_MIN_DIM = json_config['IMAGE_MIN_DIM']
        self.IMAGE_MAX_DIM = json_config['IMAGE_MAX_DIM']
        self.IMAGE_MIN_SCALE = json_config['IMAGE_MIN_SCALE']
        self.IMAGE_CHANNEL_COUNT = json_config['IMAGE_CHANNEL_COUNT']
        self.MEAN_PIXEL = np.array(json_config['MEAN_PIXEL'])
        self.TRAIN_ROIS_PER_IMAGE = json_config['TRAIN_ROIS_PER_IMAGE']
        self.ROI_POSITIVE_RATIO = json_config['ROI_POSITIVE_RATIO']
        self.POOL_SIZE = json_config['POOL_SIZE']
        self.MASK_POOL_SIZE = json_config['MASK_POOL_SIZE']
        self.MASK_SHAPE = json_config['MASK_SHAPE']
        self.MAX_GT_INSTANCES = json_config['MAX_GT_INSTANCES']
        self.RPN_BBOX_STD_DEV = np.array(json_config['RPN_BBOX_STD_DEV'])
        self.BBOX_STD_DEV = np.array(json_config['BBOX_STD_DEV'])
        self.DETECTION_MAX_INSTANCES = json_config['DETECTION_MAX_INSTANCES']
        self.DETECTION_MIN_CONFIDENCE = json_config['DETECTION_MIN_CONFIDENCE']
        self.DETECTION_NMS_THRESHOLD =json_config['DETECTION_NMS_THRESHOLD']
        self.LEARNING_RATE = json_config['LEARNING_RATE']
        self.LEARNING_MOMENTUM = json_config['LEARNING_MOMENTUM']
        self.WEIGHT_DECAY =json_config['WEIGHT_DECAY']
        self.LOSS_WEIGHTS = json_config['LOSS_WEIGHTS']
        self.USE_RPN_ROIS = json_config['USE_RPN_ROIS']
        self.TRAIN_BN = json_config['TRAIN_BN']
        self.GRADIENT_CLIP_NORM = json_config['GRADIENT_CLIP_NORM']
        self.BATCH_SIZE = json_config['BATCH_SIZE'] 
        self.IMAGE_SHAPE = json_config['IMAGE_SHAPE'] 
        self.IMAGE_META_SIZE = json_config['IMAGE_META_SIZE']

    """Base configuration class. For custom configurations, create a
        sub-class that inherits from this one and override properties
        that need to be changed.
        """
    @staticmethod
    def create_json(count_classes, count_image_per_gpu = 1, count_gpu = 1):
        config = {}

        # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
        config['GPU_COUNT'] = count_gpu

        # Number of images to train with on each GPU. A 12GB GPU can typically
        # handle 2 images of 1024x1024px.
        # Adjust based on your GPU memory and image sizes. Use the highest
        # number that your GPU can handle for best performance.
        config['IMAGES_PER_GPU'] = count_image_per_gpu

        # Number of classification classes (including background)
        config['NUM_CLASSES'] = count_classes 

        # Number of training steps per epoch
        # This doesn't need to match the size of the training set. Tensorboard
        # updates are saved at the end of each epoch, so setting this to a
        # smaller number means getting more frequent TensorBoard updates.
        # Validation stats are also calculated at each epoch end and they
        # might take a while, so don't set this too small to avoid spending
        # a lot of time on validation stats.
        config['STEPS_PER_EPOCH'] = 1000

        # Number of validation steps to run at the end of every training epoch.
        # A bigger number improves accuracy of validation stats, but slows
        # down the training.
        config['VALIDATION_STEPS'] = 50

        # Backbone network architecture
        # Supported values are: resnet50, resnet101.
        # You can also provide a callable that should have the signature
        # of model.resnet_graph. If you do so, you need to supply a callable
        # to COMPUTE_BACKBONE_SHAPE as well
        config['BACKBONE'] = "resnet101"

        # Only useful if you supply a callable to BACKBONE. Should compute
        # the shape of each layer of the FPN Pyramid.
        # See model.compute_backbone_shapes
        config['COMPUTE_BACKBONE_SHAPE'] = None

        # The strides of each layer of the FPN Pyramid. These values
        # are based on a Resnet101 backbone.
        config['BACKBONE_STRIDES'] = [4, 8, 16, 32, 64]

        # Size of the fully-connected layers in the classification graph
        config['FPN_CLASSIF_FC_LAYERS_SIZE'] = 1024

        # Size of the top-down layers used to build the feature pyramid
        config['TOP_DOWN_PYRAMID_SIZE'] = 256

    

        # Length of square anchor side in pixels
        config['RPN_ANCHOR_SCALES'] = [32, 64, 128, 256, 512]

        # Ratios of anchors at each cell (width/height)
        # A value of 1 represents a square anchor, and 0.5 is a wide anchor
        config['RPN_ANCHOR_RATIOS'] = [0.5, 1, 2]

        # Anchor stride
        # If 1 then anchors are created for each cell in the backbone feature map.
        # If 2, then anchors are created for every other cell, and so on.
        config['RPN_ANCHOR_STRIDE'] = 1

        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        config['RPN_NMS_THRESHOLD'] = 0.7

        # How many anchors per image to use for RPN training
        config['RPN_TRAIN_ANCHORS_PER_IMAGE'] = 256
        
        # ROIs kept after tf.nn.top_k and before non-maximum suppression
        config['PRE_NMS_LIMIT'] = 6000

        # ROIs kept after non-maximum suppression (training and inference)
        config['POST_NMS_ROIS_TRAINING'] = 2000
        config['POST_NMS_ROIS_INFERENCE'] = 1000

        # If enabled, resizes instance masks to a smaller size to reduce
        # memory load. Recommended when using high-resolution images.
        config['USE_MINI_MASK'] = True #TODO are these even still used? since we always scale to 1024x1024??
        config['MINI_MASK_SHAPE'] = [56, 56]  # (height, width) of the mini-mask
        
        # Input image resizing
        # Generally, use the "square" resizing mode for training and predicting
        # and it should work well in most cases. In this mode, images are scaled
        # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
        # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
        # padded with zeros to make it a square so multiple images can be put
        # in one batch.
        # Available resizing modes:
        # none:   No resizing or padding. Return the image unchanged.
        # square: Resize and pad with zeros to get a square image
        #         of size [max_dim, max_dim].
        # pad64:  Pads width and height with zeros to make them multiples of 64.
        #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
        #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
        #         The multiple of 64 is needed to ensure smooth scaling of feature
        #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
        # crop:   Picks random crops from the image. First, scales the image based
        #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
        #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
        #         IMAGE_MAX_DIM is not used in this mode.
        config['IMAGE_RESIZE_MODE'] = "square"
        config['IMAGE_MIN_DIM'] = 800
        config['IMAGE_MAX_DIM'] = 1024
        # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
        # up scaling. For example, if set to 2 then images are scaled up to double
        # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
        # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
        config['IMAGE_MIN_SCALE'] = 0
        # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
        # Changing this requires other changes in the code. See the WIKI for more
        # details: https://github.com/matterport/Mask_RCNN/wiki
        config['IMAGE_CHANNEL_COUNT'] = 3

        # Image mean (RGB)
        config['MEAN_PIXEL'] = [123.7, 116.8, 103.9]

        # Number of ROIs per image to feed to classifier/mask heads
        # The Mask RCNN paper uses 512 but often the RPN doesn't generate
        # enough positive proposals to fill this and keep a positive:negative
        # ratio of 1:3. You can increase the number of proposals by adjusting
        # the RPN NMS threshold.
        config['TRAIN_ROIS_PER_IMAGE'] = 200

        # Percent of positive ROIs used to train classifier/mask heads
        config['ROI_POSITIVE_RATIO'] = 0.33

        # Pooled ROIs
        config['POOL_SIZE'] = 7
        config['MASK_POOL_SIZE'] = 14

        # Shape of output mask
        # To change this you also need to change the neural network mask branch
        config['MASK_SHAPE'] = [28, 28]

        # Maximum number of ground truth instances to use in one image
        config['MAX_GT_INSTANCES'] = 100

        # Bounding box refinement standard deviation for RPN and final detections.
        config['RPN_BBOX_STD_DEV'] = [0.1, 0.1, 0.2, 0.2]
        config['BBOX_STD_DEV'] = [0.1, 0.1, 0.2, 0.2]

        # Max number of final detections
        config['DETECTION_MAX_INSTANCES'] = 100

        # Minimum probability value to accept a detected instance
        # ROIs below this threshold are skipped
        config['DETECTION_MIN_CONFIDENCE'] = 0.7

        # Non-maximum suppression threshold for detection
        config['DETECTION_NMS_THRESHOLD'] = 0.3

        # Learning rate and momentum
        # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
        # weights to explode. Likely due to differences in optimizer
        # implementation.
        config['LEARNING_RATE'] = 0.001
        config['LEARNING_MOMENTUM'] = 0.9

        # Weight decay regularization
        config['WEIGHT_DECAY'] = 0.0001

        # Loss weights for more precise optimization.
        # Can be used for R-CNN training setup.
        config['LOSS_WEIGHTS'] = {
            "rpn_class_loss": 1.,
            "rpn_bbox_loss": 1.,
            "mrcnn_class_loss": 1.,
            "mrcnn_bbox_loss": 1.,
            "mrcnn_mask_loss": 1.
        }

        # Use RPN ROIs or externally generated ROIs for training
        # Keep this True for most situations. Set to False if you want to train
        # the head branches on ROI generated by code rather than the ROIs from
        # the RPN. For example, to debug the classifier head without having to
        # train the RPN.
        config['USE_RPN_ROIS'] = True

        # Train or freeze batch normalization layers
        #     None: Train BN layers. This is the normal mode
        #     False: Freeze BN layers. Good when using a small batch size
        #     True: (don't use). Set layer in training mode even when predicting
        config['TRAIN_BN'] = False  # Defaulting to False since batch size is often small

        # Gradient norm clipping
        config['GRADIENT_CLIP_NORM'] = 5.0

        #processing
        config['BATCH_SIZE'] = config['IMAGES_PER_GPU'] * config['GPU_COUNT']
        if config['IMAGE_RESIZE_MODE'] == "crop":
            config['IMAGE_SHAPE'] = [config['IMAGE_MIN_DIM'], config['IMAGE_MIN_DIM'], config['IMAGE_CHANNEL_COUNT']]
        else:
            config['IMAGE_SHAPE']  = [config['IMAGE_MAX_DIM'], config['IMAGE_MAX_DIM'], config['IMAGE_CHANNEL_COUNT']]

        config['IMAGE_META_SIZE'] = 1 + 3 + 3 + 4 + 1 + config['NUM_CLASSES']
        return config