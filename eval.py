import os
import numpy as np
import torch
import torch.optim as optim
import dataset_util as dsutil
import unet
import cv2
import math


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def read_img(img_path,
             num_bits=10,
             crop_height=512, crop_width=1024, dataset='g2d'):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        path = img_path.format(gate_id)
        assert os.path.exists(path), "No such file : %s" % path
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[((img.shape[0] - crop_height) // 2):((img.shape[0] + crop_height) // 2),
              ((img.shape[1] - crop_width) // 2):((img.shape[1] + crop_width) // 2)]
        img = img.copy()
        img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    return img


def threshold(y1, y2, thr=1.25):
    max_ratio = np.maximum(y1 / y2, y2 / y1)
    return np.mean(max_ratio < thr, dtype=np.float64) * 100.


def rmse(y1, y2):
    diff = y1 - y2
    return math.sqrt(np.mean(diff * diff, dtype=np.float64))


def ard(y1, y2):
    return np.mean(np.abs(y1 - y2) / y2, dtype=np.float64)


def mae(y1, y2):
    return np.mean(np.abs(y1 - y2), dtype=np.float64)


def result(output, input, lidar_mask):
    output = output[lidar_mask].cpu().detach().numpy()
    input = input[lidar_mask].cpu().detach().numpy()
    output = np.clip(output * 150, 3, 80)
    input = np.clip(input * 150, 3, 80)
    Rmse = rmse(output, input)
    Ard = ard(output, input)
    Mae = mae(output, input)
    Threshold1 = threshold(output, input, thr=1.25)
    Threshold2 = threshold(output, input, thr=1.25 ** 2)
    Threshold3 = threshold(output, input, thr=1.25 ** 3)

    return Rmse, Ard, Mae, Threshold1, Threshold2, Threshold3


def eval(models, train_filenames):
    models["Encoder"].eval()
    models["Decoder"].eval()
    max_distance = 150
    min_distance = 3
    img_id = {}
    base_dir = '/home/edric/PycharmProjects/UnetWithVIT/data/real'
    Results = []
    for i in range(0, len(train_filenames)):
        print("{0}/{1}".format(i, len(train_filenames)))
        img_id[i] = train_filenames[i].split('\n')
        id = img_id[i][0]
        gate_dir = os.path.join(base_dir, 'gated{}_10bit', '{}.png'.format(id))
        in_img = read_img(gate_dir)
        in_img = torch.tensor(in_img).unsqueeze(0).to(device=device)
        in_img = in_img.permute(0, 3, 1, 2)
        input, lidar_mask = dsutil.read_gt_image(base_dir=base_dir, gta_pass='', img_id=img_id[i][0], data_type='real',
                                                 min_distance=min_distance, max_distance=max_distance)
        input = torch.tensor(input).to(device=device)
        input = input.permute(0, 3, 1, 2)
        lidar_mask = torch.tensor(lidar_mask).permute(0, 3, 1, 2)
        output = models["Decoder"](models["Encoder"](in_img))
        Result = result(output["output", 0], input, lidar_mask)
        Results.append(Result)
    res = np.array(Results).mean(0)
    metrix = "rmse={}  ard={}  mae={} delta1={}  delta2={}  delta3={}".format(
        res[0], res[1], res[2], res[3], res[4], res[5])
    print(metrix)


if __name__ == '__main__':
    data_fpath = 'splits/'
    # 数据加载
    fpath = os.path.join(data_fpath, "real_{}_night.txt")
    train_filenames = readlines(fpath.format("test"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    models["Encoder"] = unet.Encoder()
    models["Decoder"] = unet.Decoder()
    Encoderpath = "models/Encoder01.pth"
    Decoderpath = "models/Decoder01.pth"
    models["Encoder"].load_state_dict(torch.load(Encoderpath, map_location=lambda storage, loc: storage))
    models["Decoder"].load_state_dict(torch.load(Decoderpath, map_location=lambda storage, loc: storage))
    models["Encoder"].to(device=device)
    models["Decoder"].to(device=device)
    eval(models, train_filenames)
