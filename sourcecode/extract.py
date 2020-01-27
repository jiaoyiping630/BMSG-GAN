def main():
    import os
    import numpy as np
    import torch as th
    from torch.backends import cudnn
    cudnn.benchmark = True
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    from pinglib.files import get_file_list, create_dir
    from pinglib.utils import save_variables
    from PIL import Image

    image_folder = r"D:\Projects\anomaly_detection\datasets\Camelyon\test_negative"
    save_path = r"D:\Projects\anomaly_detection\BMSG_GAN_test_neg.pkl"

    '''-----------------建立数据集和数据载入器----------------'''

    from torch.utils.data import Dataset
    from torchvision.transforms import ToTensor, Resize, Compose, Normalize

    class Dataset4extract(Dataset):
        def __init__(self, image_paths):
            self.image_paths = image_paths
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx])

            img = self.transform(img)

            if img.shape[0] == 4:
                # ignore the alpha channel
                # in the image if it exists
                img = img[:3, :, :]
            return img

    image_paths = get_file_list(image_folder, ext='jpg')
    dataset = Dataset4extract(image_paths)
    print("Total number of images in the dataset:", len(dataset))

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    '''-----------------建立模型----------------'''
    from MSG_GAN.GAN import MSG_GAN
    depth = 7
    msg_gan = MSG_GAN(depth=depth,
                      latent_size=512,
                      use_eql=True,
                      use_ema=True,
                      ema_decay=0.999,
                      device=device)

    '''-----------------进行评估----------------'''
    features = []
    from torch.nn.functional import avg_pool2d

    for (i, batch) in enumerate(dataloader):
        #   获取多分辨率的图像输入
        images = batch.to(device)

        images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                             for i in range(1, depth)]
        images = list(reversed(images))

        #   把这些图像丢给模型
        feature = msg_gan.extract(images)
        features.append(feature.detach().cpu().numpy())

    '''-----------------保存结果----------------'''
    features = np.concatenate(features, axis=0)
    save_variables([features], save_path)
