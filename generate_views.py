# ref : https://github.com/kwea123/nerf_pl/blob/master/eval.py
import torch
import torch.nn as nn
from rendering_utils import run_one_iter_of_nerf
import imageio
import os
import time


def generate_views(test_ds,
                   num_coarse_sample,
                   t_i_c_bin_edges,
                   t_i_c_gap,
                   test_os,
                   chunk_size,
                   coarse_mlp,
                   num_fine_sample,
                   t_f,
                   fine_mlp,
                   dataset):
    imgs = []
    # psnr = []
    # criterion = nn.MSELoss()
    fine_mlp.eval()
    for i in range(dataset):
        with torch.no_grad():
            (_, C_rs_f) = run_one_iter_of_nerf(
                test_ds,
                num_coarse_sample,
                t_i_c_bin_edges,
                t_i_c_gap,
                test_os,
                chunk_size,
                coarse_mlp,
                num_fine_sample,
                t_f,
                fine_mlp,
            )

        # loss = criterion(C_rs_f, test_img)
        # print(f"Loss: {loss.item()}")
        imgs += C_rs_f.detach().cpu().numpy()
        # psnr += (-10.0 * torch.log10(loss)).item()

    dir_name = os.getcwd()
    imageio.mimsave(os.path.join(dir_name, f'novel_view_{time.time()}.gif'), imgs, fps=30)
    print("GIF file saved !")

