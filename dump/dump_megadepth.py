import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset
from KLM.superpoint import SuperPoint
from dump.utils import match_from_projection_points_torch

parse = argparse.ArgumentParser(description='Megadepth', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse.add_argument('--feature_type', type=str, default='sift')
parse.add_argument('--base_path', type=str, required=True)
parse.add_argument('--save_path', type=str, required=True)


def process_keypoint(all_keypoints, keypoint_dir, img_path, feature_type):
    kpt_fn = osp.join(keypoint_dir, img_path.split('/')[-1] + '_' + feature_type + '.npy')
    if kpt_fn in all_keypoints.keys():
        data = all_keypoints[kpt_fn]
    else:
        data = np.load(kpt_fn, allow_pickle=True).item()
        all_keypoints[kpt_fn] = data
    kpts = data['keypoints']
    depth = data['depth']
    if kpts.shape[0] < 1024:
        return {'skip': True}

    full_ids = np.array([v for v in range(kpts.shape[0])])
    valid_kpt_ids = (depth > 0)
    valid_kpts = kpts[valid_kpt_ids]
    valid_depth = depth[valid_kpt_ids]
    valid_ids = full_ids[valid_kpt_ids]
    if valid_ids.shape[0] <= 20:
        return {'skip': True}
    return {'valid_kpts': valid_kpts, 'valid_depth': valid_depth, 'valid_ids': valid_ids, 'skip': False}


class Megadepth(Dataset):
    def __init__(self,
                 scene_info_path,
                 base_path,
                 scene_list_fn,
                 min_overlap_ratio=0.1,
                 max_overlap_ratio=0.7,
                 max_scale_ratio=np.inf,
                 nfeatures=4096,
                 feature_type='sift',
                 save_path_keypoint =''
                 ):

        self.scene_info_path = scene_info_path
        self.base_path = base_path
        self.scene_list_fn = scene_list_fn
        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio
        self.nfeatures = nfeatures
        self.feature_type = feature_type
        self.save_path_keypoint = save_path_keypoint

        self.scenes = []
        with open(scene_list_fn, 'r') as f:
            ls = f.readlines()
            for l in ls:
                self.scenes.append(l.strip())
        print('Load images from {:d} scenes'.format(len(self.scenes)))

        if feature_type == 'sift':
            self.sift = cv2.SIFT_create(nfeatures=self.nfeatures, contrastThreshold=0.04)
        elif feature_type == 'spp':
            spp_config = {
                'descriptor_dim': 256,
                'nms_radius': 3,
                'keypoint_threshold': 0.0003,
                'max_keypoints': self.nfeatures,
                'remove_borders': 4,
                'weight_path': './weights/superpoint_v1.pth',
                'with_compensate': True,
            }
            self.spp = SuperPoint(config=spp_config).eval().cuda()

        self.image_paths = []
        self.depth_paths = []
        self.poses = []
        self.intrinsics = []
        # Find all the images with
        # image_path depth_path pose intrinsics information
        self.extract_image_fns()

    def detect_and_compute(self, img):
        if self.feature_type == 'sift':
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            cv_kp, desc = self.sift.detectAndCompute(img_gray, None)
            kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.response] for _kp in cv_kp])  # N*3
            index = np.flip(np.argsort(kp[:, 2]))
            kp, desc = kp[index], desc[index]
            descs = np.sqrt(abs(desc / (np.linalg.norm(desc, axis=-1, ord=1)[:, np.newaxis] + 1e-8)))
            kps = kp[:, 0:2]
            scores = kp[:, 2].reshape(-1, )

        elif self.feature_type == 'spp':
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            norm_img = img_gray.astype(float) / 255.
            with torch.no_grad():
                norm_img = torch.from_numpy(norm_img[None, None]).cuda().float()
                outputs = self.spp({'image': norm_img})
                kps = torch.vstack(outputs['keypoints']).cpu().numpy()
                descs = torch.vstack(outputs['descriptors']).cpu().numpy().transpose()
                scores = torch.vstack(outputs['scores']).cpu().numpy().reshape(-1, )
        return kps, scores, descs

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        pose = self.poses[idx]
        intrinsic = self.intrinsics[idx]
        image_paths = image_path.split('/')
        scene = image_paths[1]
        img_fn = image_paths[-1]
        save_fn = osp.join(self.save_path_keypoint, scene, img_fn + '_{:s}.npy'.format(self.feature_type))
        if osp.exists(save_fn):
            return {"skip": True}

        with h5py.File(osp.join(self.base_path, depth_path), 'r') as hf5_f:
            depth = np.array(hf5_f['/depth'])
            assert (np.min(depth) >= 0)

        image = cv2.imread(osp.join(self.base_path, image_path))
        kpts, scores, descs = self.detect_and_compute(img=image)
        depth_values = depth[kpts.astype(int)[:, 1], kpts.astype(int)[:, 0]]

        out = {
            'keypoints': kpts,
            'scores': scores,
            'descriptors': descs,
            'image_size': np.array(image.shape, int),
            'depth': depth_values,
            'image_path': image_path,
            'depth_path': depth_path,
            'pose': pose,
            'intrinsics': intrinsic,
            'skip': False
        }

        return out

    def __len__(self):
        return len(self.image_paths)

    def build_correspondence(self, scene, save_dir, pre_load=True):
        keypoint_dir = osp.join(save_dir, 'keypoints_{:s}'.format(self.feature_type), scene)
        match_dir = osp.join(save_dir, 'matches_{:s}'.format(self.feature_type))
        nmatches_info_dir = osp.join(save_dir, 'nmatches_{:s}'.format(self.feature_type))  # only for random sampling
        os.makedirs(match_dir, exist_ok=True)
        os.makedirs(nmatches_info_dir, exist_ok=True)

        if osp.exists(osp.join(match_dir, scene + '.npy')):
            print('{:s} exist'.format(scene))
            return None

        scene_info_path = osp.join(self.scene_info_path, '{:s}.0.npz'.format(scene))
        scene_info = np.load(scene_info_path, allow_pickle=True)
        overlap_matrix = scene_info['overlap_matrix']
        scale_ratio_matrix = scene_info['scale_ratio_matrix']

        valid = np.logical_and(
            np.logical_and(overlap_matrix >= self.min_overlap_ratio,
                           overlap_matrix <= self.max_overlap_ratio),
            scale_ratio_matrix <= self.max_scale_ratio
        )
        pairs = np.vstack(np.where(valid))
        selected_ids = np.arange(0, pairs.shape[1])
        print('Find {:d} pairs from scene {:s}'.format(len(selected_ids), scene))

        image_paths = scene_info['image_paths']
        depth_paths = scene_info['depth_paths']
        points3D_id_to_2D = scene_info['points3D_id_to_2D']
        intrinsics = scene_info['intrinsics']
        poses = scene_info['poses']
        valid_pairs = []

        all_keypoints = {}
        if pre_load:
            print('Loading keypoints...')
            for img_path in tqdm(image_paths, total=len(image_paths)):
                if img_path is None:
                    continue
                kpt_fn = osp.join(keypoint_dir, img_path.split('/')[-1] + '_' + self.feature_type + '.npy')
                if osp.isfile(kpt_fn):
                    data = np.load(kpt_fn, allow_pickle=True).item()
                    all_keypoints[kpt_fn] = data

        for pair_idx in tqdm(selected_ids, total=len(selected_ids)):
            idx1, idx2 = pairs[0, pair_idx], pairs[1, pair_idx]

            matches = np.array(list(
                points3D_id_to_2D[idx1].keys() &
                points3D_id_to_2D[idx2].keys()
            ))
            if len(matches) < 20:
                continue

            # conditions for rejecting pairs: number of spp points, image size
            image_path1, image_path2 = image_paths[idx1], image_paths[idx2]
            pose1, pose2 = poses[idx1], poses[idx2]
            intrinsics1, intrinsics2 = intrinsics[idx1], intrinsics[idx2]
            ret1 = process_keypoint(all_keypoints, keypoint_dir, image_path1, self.feature_type)
            ret2 = process_keypoint(all_keypoints, keypoint_dir, image_path2, self.feature_type)
            if ret1['skip'] or ret2['skip']:
                continue
            valid_kpts1, valid_kpts2 = ret1['valid_kpts'], ret2['valid_kpts']
            valid_depth1, valid_depth2 = ret1['valid_depth'], ret2['valid_depth']
            valid_ids1, valid_ids2 = ret1['valid_ids'], ret2['valid_ids']

            with torch.no_grad():
                inlier_matches, outlier_matches = match_from_projection_points_torch(
                    pos1=torch.from_numpy(valid_kpts1.transpose()).float().cuda(),
                    depth1=torch.from_numpy(valid_depth1).float().cuda(),
                    intrinsics1=torch.from_numpy(intrinsics1).float().cuda(),
                    pose1=torch.from_numpy(pose1).float().cuda(),
                    bbox1=None,
                    pos2=torch.from_numpy(valid_kpts2.transpose()).float().cuda(),
                    depth2=torch.from_numpy(valid_depth2).float().cuda(),
                    intrinsics2=torch.from_numpy(intrinsics2).float().cuda(),
                    pose2=torch.from_numpy(pose2).float().cuda(),
                    bbox2=None,
                    inlier_th=5, outlier_th=15, cycle_check=True,
                )
                inlier_matches = inlier_matches.cpu().numpy()

            if inlier_matches.shape[0] <= 20:
                continue

            matched_ids1, matched_ids2 = [], []
            for m in inlier_matches:
                if valid_ids1[m[0]] in matched_ids1 or valid_ids2[m[1]] in matched_ids2:
                    continue
                matched_ids1.append(valid_ids1[m[0]])
                matched_ids2.append(valid_ids2[m[1]])

            valid_pairs.append({
                'image_path1': image_paths[idx1],
                'depth_path1': depth_paths[idx1],
                'intrinsics1': intrinsics[idx1],
                'pose1': poses[idx1],
                'image_path2': image_paths[idx2],
                'depth_path2': depth_paths[idx2],
                'intrinsics2': intrinsics[idx2],
                'pose2': poses[idx2],
                'matched_ids1': np.array(matched_ids1, dtype=int),
                'matched_ids2': np.array(matched_ids2, dtype=int),
            })

        if len(valid_pairs) > 0:
            np.save(osp.join(match_dir, scene), valid_pairs)

        print('Find {:d}/{:d} valid pairs from scene {:s}'.format(len(valid_pairs), len(selected_ids), scene))
        scene_nvalid = {scene: len(valid_pairs)}
        np.save(osp.join(save_dir, nmatches_info_dir, '{:s}_{:s}'.format(scene, self.feature_type)), scene_nvalid)
        del all_keypoints

    def write_matches(self, save_dir, scene_list):
        match_dir = osp.join(save_dir, 'matches_{:s}'.format(self.feature_type))
        save_root = osp.join(save_dir, 'matches_sep_{:s}'.format(self.feature_type))
        for fn in tqdm(scene_list, total=len(scene_list)):
            npy_path = osp.join(match_dir, fn + ".npy")
            if not osp.isfile(npy_path):
                continue
            data = np.load(npy_path, allow_pickle=True)
            save_dir_scene = osp.join(save_root, fn.split('.')[0])
            if not osp.exists(save_dir_scene):
                os.makedirs(save_dir_scene)
            for idx, d in tqdm(enumerate(data), total=len(data)):
                np.save(osp.join(save_dir_scene, '{:d}'.format(idx)), d)

    def extract_image_fns(self):
        for scene in tqdm(self.scenes, total=len(self.scenes)):
            scene_info_path = osp.join(self.scene_info_path, '{:s}.0.npz'.format(scene))
            if not osp.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            # Modification to adjust our method
            for i in range(len(depth_paths)):
                if depth_paths[i] is not None:
                    depth_paths[i] = depth_paths[i].replace("phoenix/S6/zl548/MegaDepth_v1/", "depth_undistorted/")
                    depth_paths[i] = depth_paths[i].replace("dense0/depths/", "")
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']

            assert len(image_paths) == len(depth_paths)
            assert len(image_paths) == len(intrinsics)
            assert len(image_paths) == len(poses)

            for ni in range(len(image_paths)):
                image_path = image_paths[ni]
                depth_path = depth_paths[ni]
                pose = poses[ni]
                intrinsic = intrinsics[ni]
                if None not in [image_path, depth_path, pose, intrinsic]:
                    self.image_paths.append(image_path)
                    self.depth_paths.append(depth_path)
                    self.poses.append(pose)
                    self.intrinsics.append(intrinsic)
        print('Find {:d} images in total'.format(len(self.image_paths)))


if __name__ == '__main__':
    args = parse.parse_args()
    feat_type = args.feature_type
    base_path = args.base_path
    save_path = args.save_path
    scene_list_fn = 'assets/megadepth_scenes_full.txt'  # for training
    # scene_list_fn = 'assets/megadepth_scenes_debug.txt'  # for test only

    save_path_keypoint = osp.join(save_path, 'keypoints_{:s}'.format(feat_type))

    scenes = []
    with open(scene_list_fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            scenes.append(line.strip())

    scene_info_path = osp.join(base_path, 'scene_info')
    dataset = Megadepth(scene_info_path=scene_info_path,
                        base_path=base_path,
                        scene_list_fn=scene_list_fn,
                        nfeatures=4096,
                        feature_type=feat_type,
                        min_overlap_ratio=0.1,
                        max_overlap_ratio=0.8,
                        save_path_keypoint=save_path_keypoint,
                        )

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         num_workers=0,
                                         shuffle=False,
                                         batch_size=1,
                                         pin_memory=True,
                                         )

    print('Start extracting keypoints...')
    for bid, data in tqdm(enumerate(loader), total=len(loader)):
        skip = data['skip']
        if skip:
            continue
        keypoints = data['keypoints'][0].numpy()
        scores = data['scores'][0].numpy()
        descriptors = data['descriptors'][0].numpy()
        image_size = data['image_size'][0].numpy()
        depth = data['depth'][0].numpy()
        image_path = data['image_path'][0]
        depth_path = data['depth_path'][0]
        pose = data['pose'][0].numpy()
        intrinsics = data['intrinsics'][0].numpy()

        image_paths = image_path.split('/')
        scene = image_paths[1]
        img_fn = image_paths[-1]

        if not osp.exists(osp.join(save_path_keypoint, scene)):
            os.makedirs(osp.join(save_path_keypoint, scene))

        save_fn = osp.join(save_path_keypoint, scene, img_fn + '_{:s}'.format(feat_type))
        save_data = {
            'image_path': image_path,
            'depth_path': depth_path,
            'intrinsics': intrinsics,
            'pose': pose,
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'image_size': image_size,
            'depth': depth,
        }
        np.save(save_fn, save_data)
    print('Finish extracting keypoints...')

    print('Start building correspondences...')
    for scene in scenes:
        # you can split it into several slices and do them parallelly
        dataset.build_correspondence(scene=scene, save_dir=save_path, pre_load=True)
        dataset.write_matches(save_dir=save_path, scene_list=[scene])

    # merge scene-pairs to a single file for random sampling in the training process
    mega_scene_pairs = {}
    for scene in scenes:
        s_info_path = osp.join(save_path, 'nmatches_{:s}'.format(feat_type), '{:s}_{:s}.npy'.format(scene, feat_type))
        scene_info = np.load(s_info_path, allow_pickle=True)[()]
        mega_scene_pairs = {**mega_scene_pairs, **scene_info}
    np.save(osp.join(save_path, 'mega_scene_nmatches_{:s}'.format(feat_type)), mega_scene_pairs)
    print('Finish building correspondences...')
