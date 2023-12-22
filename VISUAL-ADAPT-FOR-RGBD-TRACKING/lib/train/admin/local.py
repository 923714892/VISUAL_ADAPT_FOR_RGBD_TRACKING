class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/zbn/data/zgt/CVPR2024/ViPT-main/ViPT-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/zbn/data/zgt/CVPR2024/ViPT-main/ViPT-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/zbn/data/zgt/CVPR2024/ViPT-main/ViPT-main/pretrained_networks'
        self.got10k_val_dir = '/media/zbn/data/CVPR2024/ViPT-main/ViPT-main/data/got10k/val'
        self.lasot_lmdb_dir = '/media/zbn/data/zgt/CVPR2024/ViPT-main/ViPT-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/local_data/zgt/CVPR2024/ViPT-main/ViPT-main/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/home/local_data/zgt/CVPR2024/ViPT-main/ViPT-main/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/home/local_data/zgt/CVPR2024/ViPT-main/ViPT-main/data/coco_lmdb'
        self.coco_dir = '/media/zbn/data/zgt/CVPR2024/ViPT-main/ViPT-main/data/coco'
        self.lasot_dir = '/media/zbn/data/zgt/CVPR2024/ViPT-main/ViPT-main/data/lasot'
        self.got10k_dir = '/home/local_data/zgt/CVPR2024/ViPT-main/ViPT-main/data/got10k/train'
        self.trackingnet_dir = '/media/zbn/data/zgt/CVPR2024/ViPT-main/ViPT-main/data/trackingnet'
        self.depthtrack_dir = '/media/zbn/data/hxt/data/DethTrack/train'
        self.lasher_dir = '/media/zbn/data/zgt/data/LasHeR/TrainingSet/trainingset'
        self.visevent_dir = '/media/zbn/data/zgt/data/VisEvent_dataset/train_subset'
