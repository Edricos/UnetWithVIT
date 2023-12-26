import os
import numpy as np
import torch
import torch.optim as optim
import dataset_util as dsutil
import unet
import cv2
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def Loss(output, input, lidar_mask, in_img):
    dis = output - input
    l1_loss = dis[lidar_mask].abs().mean()
    # grad_loss = tv_loss(in_img,output)
    return l1_loss


def tv_loss(in_img, output):
    input = torch.mean(in_img.requires_grad_(True), dim=1)
    input_x, input_y = gradient(input.unsqueeze(0))
    output_x, output_y = gradient(output)
    input_x = input_x.abs()
    input_y = input_y.abs()
    output_x = output_x.abs()
    output_y = output_y.abs()
    ep_dy = torch.exp(-input_y)
    ep_dx = torch.exp(-input_x)
    grad_loss = torch.mean(ep_dy * output_y + ep_dx * output_x)
    return grad_loss


def gradient(x):
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


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


def train(models, train_filenames):
    models["Encoder"].train()
    models["Decoder"].train()
    max_distance = 150
    min_distance = 3
    img_id = {}
    base_dir = '/home/edric/PycharmProjects/UnetWithVIT/data/real'

    gta_pass = ''
    model_optimizer_Encoder = optim.Adam(models["Encoder"].parameters(), lr=1e-4)
    model_optimizer_Decoder = optim.Adam(models["Decoder"].parameters(), lr=1e-4)
    for epoch in range(2):
        for i in tqdm.tqdm(range(0, len(train_filenames))):
            img_id[i] = train_filenames[i].split('\n')
            id = img_id[i][0]
            gate_dir = os.path.join(base_dir, 'gated{}_10bit', '{}.png'.format(id))
            in_img = dsutil.read_gated_image(base_dir=base_dir, gta_pass=gta_pass,
                                             img_id=img_id[i][0], data_type='real')
            in_img = torch.tensor(in_img).to(device=device)
            in_img = in_img.permute(0, 3, 1, 2)
            input, lidar_mask = dsutil.read_gt_image(base_dir=base_dir, gta_pass=gta_pass,
                                                     img_id=img_id[i][0], data_type='real',
                                                     min_distance=min_distance, max_distance=max_distance)
            input = torch.tensor(input).to(device=device)
            input = input.permute(0, 3, 1, 2)
            lidar_mask = torch.tensor(lidar_mask).permute(0, 3, 1, 2)
            output = models["Decoder"](models["Encoder"](in_img))



            loss = Loss(output["output", 0], input, lidar_mask, in_img)
            model_optimizer_Encoder.zero_grad()
            model_optimizer_Decoder.zero_grad()
            loss.backward()
            model_optimizer_Encoder.step()
            model_optimizer_Decoder.step()
            # tqdm.tqdm.set_postfix('epoch:{0}  [{1}/{2}]  loss:{3}'.format(epoch, i + 1, len(train_filenames), loss))
            # print('epoch:{0}  [{1}/{2}]  loss:{3}'.format(epoch, i + 1, len(train_filenames), loss))
    return models


if __name__ == '__main__':
    data_fpath = 'splits/data'
    # 数据加载
    fpath = os.path.join(data_fpath, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    models = {}
    models["Encoder"] = unet.Encoder()
    models["Decoder"] = unet.Decoder()
    models["Encoder"].to(device=device)
    models["Decoder"].to(device=device)

    model = train(models, train_filenames)  # train for one epoch
    model_dir = "models/"
    torch.save(model["Encoder"].state_dict(), os.path.join(model_dir, 'Encoder01.pth'))
    torch.save(model["Decoder"].state_dict(), os.path.join(model_dir, 'Decoder01.pth'))



