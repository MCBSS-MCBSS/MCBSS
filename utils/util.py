import shutil
from torch.utils import data
import copy
import os
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.ceil(W * cut_rat).astype(int)
    cut_h = np.ceil(H * cut_rat).astype(int)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def GLMC_mixed(org1, org2, invs1, invs2, label_org, label_invs, label_org_w, label_invs_w, alpha=1):
    lam = np.random.beta(alpha, alpha)
    mixup_x = lam * org1 + (1 - lam) * invs1
    mixup_y = lam * label_org + (1 - lam) * label_invs
    mixup_y_w = lam * label_org_w + (1 - lam) * label_invs_w

    bbx1, bby1, bbx2, bby2 = rand_bbox(org2.size(), lam)
    org2[:, :, bbx1:bbx2, bby1:bby2] = invs2[:, :, bbx1:bbx2, bby1:bby2]

    lam_cutmix = lam
    cutmix_y = lam_cutmix * label_org + (1 - lam_cutmix) * label_invs
    cutmix_y_w = lam_cutmix * label_org_w + (1 - lam_cutmix) * label_invs_w

    return mixup_x, org2, mixup_y, cutmix_y, mixup_y_w, cutmix_y_w
