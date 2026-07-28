import os
import glob
import numpy as np
import cv2
import torch
import sys
from pathlib import Path
from .base_dumper import BaseDumper, np_skew_symmetric

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def load_image(path: Path, grayscale=False) -> torch.Tensor:
    image = read_image(path, grayscale=grayscale)
    return numpy_image_to_torch(image)


def parse_camera(calib_elems):
    K = np.array([float(x) for x in calib_elems[:9]]).reshape(3, 3).astype(np.float32)
    return K


def parse_relative_pose(pose_elems):
    # assert len(calib_list) == 9
    R, t = pose_elems[:9], pose_elems[9:12]
    R = np.array([float(x) for x in R]).reshape(3, 3).astype(np.float32)
    t = np.array([float(x) for x in t]).astype(np.float32)
    return R, t


class megadepth1500(BaseDumper):

    def get_seqs(self):
        data_dir = os.path.join(self.config['rawdata_dir'], 'images')
        for seq in self.config['data_seq']:
            seq_dir = os.path.join(data_dir, seq)
            dump_dir = os.path.join(self.config['feature_dump_dir'], seq)
            cur_img_seq = glob.glob(os.path.join(seq_dir, '*.jpg'))
            cur_dump_seq = [
                os.path.join(dump_dir, path.split('/')[-1]) + '_' + self.config['extractor']['name'] + '_' + str(
                    self.config['extractor']['num_kpt']) + '.hdf5' for path in cur_img_seq]
            self.img_seq += cur_img_seq
            self.dump_seq += cur_dump_seq

    def format_dump_folder(self):
        if not os.path.exists(self.config['feature_dump_dir']):
            os.mkdir(self.config['feature_dump_dir'])
        for seq in self.config['data_seq']:
            seq_dir = os.path.join(self.config['feature_dump_dir'], seq)
            if not os.path.exists(seq_dir):
                os.mkdir(seq_dir)

    def _read_view(self, name):
        path = Path(self.config['rawdata_dir'], 'images', name)
        img = load_image(path)
        return img

    def format_dump_data(self):
        print("Formatting data")
        pair_f = self.config["pairs"]
        with open(pair_f, "r") as f:
            self.pairs = [line.rstrip() for line in f]
        self.data = {'K1': [], 'K2': [], 'R': [], 'T': [], 'e': [], 'f': [],
                     'fea_path1': [], 'fea_path2': [],
                     'img_path1': [], 'img_path2': []}

        for cur_pair in self.pairs:
            pair_data = cur_pair.split(" ")
            name0, name1 = pair_data[:2]
            K1 = parse_camera(pair_data[2:11])
            K2 = parse_camera(pair_data[11:20])
            dR, dt = parse_relative_pose(pair_data[20:32])
            dt /= np.sqrt(np.sum(dt ** 2))
            e_gt_unnorm = np.reshape(np.matmul(
                np.reshape(np_skew_symmetric(dt.astype('float64').reshape(1, 3)), (3, 3)),
                np.reshape(dR.astype('float64'), (3, 3))), (3, 3))
            e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)
            f_gt_unnorm = np.linalg.inv(K2.T) @ e_gt @ np.linalg.inv(K1)
            f_gt = f_gt_unnorm / np.linalg.norm(f_gt_unnorm)

            self.data['K1'].append(K1), self.data['K2'].append(K2)
            self.data['R'].append(dR), self.data['T'].append(dt)
            self.data['e'].append(e_gt), self.data['f'].append(f_gt)

            self.data['img_path1'].append(name0), self.data['img_path2'].append(name1)
            dump_seq_dir = os.path.join(self.config['feature_dump_dir'])
            fea_path1 = os.path.join(dump_seq_dir, name0.split('/')[0],
                                     name0.split('/')[-1] + '_' + self.config['extractor']['name']
                                     + '_' + str(self.config['extractor']['num_kpt']) + '.hdf5')

            fea_path2 = os.path.join(dump_seq_dir, name1.split('/')[0],
                                     name1.split('/')[-1] + '_' + self.config['extractor']['name']
                                     + '_' + str(self.config['extractor']['num_kpt']) + '.hdf5')
            self.data['fea_path1'].append(fea_path1), self.data['fea_path2'].append(fea_path2)

        self.form_standard_dataset()
