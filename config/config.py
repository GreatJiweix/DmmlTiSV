class Config(object):

    num_classes = 5994
    easy_margin = False
    use_se = False

    logs_root = "/home/xjw/workspace/DmmlTiSV-master/logs/"
    data_dir = "/data1/xjw/vox1/"
    noise_dir = "/data2/xjw/RIRS_NOISES/pointsource_noises/"
    noise_files = "/data2/xjw/RIRS_NOISES/noise.scp"

    vox_root = "/data1/xjw/vox1_test/"
    vox_test_list = "/home/xjw/workspace/DmmlTiSV-master/veri_test.txt"
    vox_val_list = "/home/xjw/workspace/triplet-network-pytorch-master/veri_little.txt"

    ims_ids = 64
    ims_per_id = 2
    degree = 45

    use_gpu = True  # use GPU or not

    num_workers = 8  # how many workers for loading data

    max_epoch = 500
    lr = 3e-4  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
