import torch


def reproject_points_torch(pos1, depth1, intrinsics1, pose1, bbox1, intrinsics2, pose2, bbox2):
    Z1 = depth1

    # COLMAP convention
    if bbox1 is not None:
        u1 = pos1[0, :] + bbox1[1] + .5
        v1 = pos1[1, :] + bbox1[0] + .5
    else:
        u1 = pos1[0, :] + .5
        v1 = pos1[1, :] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = torch.vstack([
        X1.reshape(1, -1),
        Y1.reshape(1, -1),
        Z1.reshape(1, -1),
        torch.ones_like(Z1).reshape(1, -1),
    ])

    XYZ2_hom = (pose2.float() @ torch.linalg.inv(pose1).float()) @ XYZ1_hom.float()
    XYZ2 = XYZ2_hom[:-1, :] / (XYZ2_hom[-1, :].reshape(1, -1) + 1e-5)
    uv2_hom = intrinsics2 @ XYZ2
    uv2 = uv2_hom[:-1, :] / (uv2_hom[-1, :].reshape(1, -1) + 1e-5)

    if bbox2 is not None:
        u2 = uv2[0, :] - bbox2[1] - .5
        v2 = uv2[1, :] - bbox2[0] - .5
    else:
        u2 = uv2[0, :] - .5
        v2 = uv2[1, :] - .5
    uv2 = torch.vstack([u2.reshape(1, -1), v2.reshape(1, -1)])

    return uv2


def match_from_projection_points_torch(
        pos1, depth1, intrinsics1, pose1, bbox1,
        pos2, depth2, intrinsics2, pose2, bbox2,
        inlier_th=3, outlier_th=5,
        cycle_check=False):  # [2, M]

    proj_uv12 = reproject_points_torch(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
                                       intrinsics2=intrinsics2, pose2=pose2, bbox2=None)  # [2, N]
    N, M = pos1.shape[1], pos2.shape[1]

    proj_uv12_ext = proj_uv12[:, :, None].repeat(1, 1, M)
    pos2_ext = pos2[:, None, :].repeat(1, N, 1)
    error_uv12 = proj_uv12_ext - pos2_ext
    error_uv12 = torch.sqrt(torch.sum(error_uv12 ** 2, dim=0))

    matches_12 = torch.argmin(error_uv12, dim=1)
    errors_12 = error_uv12[torch.arange(error_uv12.shape[0]), matches_12]

    inlier_ids12 = torch.where(errors_12 <= inlier_th)[0]
    outlier_ids12 = torch.where(errors_12 >= outlier_th)[0]

    inlier_matches12 = torch.vstack([inlier_ids12, matches_12[inlier_ids12]]).long().t()  # [N, 2]
    outlier_matches12 = torch.vstack([outlier_ids12, matches_12[outlier_ids12]]).long().t()  # [N, 2]

    if not cycle_check:
        return inlier_matches12, outlier_matches12

    matched_pos1 = pos1[:, inlier_matches12[:, 0]]
    matched_pos2 = pos2[:, inlier_matches12[:, 1]]
    matched_depth2 = depth2[inlier_matches12[:, 1]]

    proj_uv21 = reproject_points_torch(pos1=matched_pos2, depth1=matched_depth2, intrinsics1=intrinsics2, pose1=pose2,
                                       bbox1=bbox2,
                                       intrinsics2=intrinsics1, pose2=pose1, bbox2=bbox1)  # [2, M]
    error_uv21 = proj_uv21 - matched_pos1
    error_uv21 = torch.sqrt(torch.sum(error_uv21 ** 2, dim=0))
    inliers21 = (error_uv21 <= inlier_th)
    outlier21 = (error_uv21 >= outlier_th)

    inlier_cycle = inlier_matches12[inliers21]
    return inlier_cycle, outlier_matches12