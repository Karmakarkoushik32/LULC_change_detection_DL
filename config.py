# Statndard image resolution and size
TARGET_RES = 0.1  # 10 cm
IMAGE_PATCH_SIZE = 512


################## TRAINING CONF #######################
#  paths
TRAIN_IMG_DIR = "datasets/training/images"
TRAIN_MASK_DIR = "datasets/training/masks"
VAL_IMG_DIR = "datasets/validation/images"
VAL_MASK_DIR = "datasets/validation/masks"
CHECKPOINT_DIR = "checkpoints"

#  model
BACKBONE = "resnet34"  # "resnet34" | "resnet50"
CLASS_MAP = {
    0: "others(ignored)",
    1: "building",
    2: "road",
    3: "waterbody",
    4: "vegetation",
}  # index 0 = others, excluded via IGNORE_INDEX
NUM_CLASSES = len(
    CLASS_MAP
)  # 0=others(ignored), 1=building, 2=road, 3=waterbody, 4=vegetation

# Class index 0 (background/unlabelled) is excluded from loss and metrics.
# Masks should use 0 for pixels you want ignored, 1-4 for the real classes.
PRETRAINED = True

#  dataset Normalization
IMAGENET_NORM_MEAN = [0.485, 0.456, 0.406]
IMAGENET_NORM_STD = [0.229, 0.224, 0.225]

#  training
EPOCHS = 5
BATCH_SIZE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0  # max gradient norm
FREEZE_EPOCHS = 5  # keep encoder frozen for first N epochs

#  scheduler
WARMUP_EPOCHS = 3
LR_MIN = LR * 0.01  # cosine decay floor

#  misc
NUM_WORKERS = 0
IGNORE_INDEX = 0  # class 0 = unlabelled/background — excluded from loss & metrics
SEED = 42
