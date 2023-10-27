import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from rendering_utils import run_one_iter_of_nerf, render_radiance_volume, get_fine_query_points, get_coarse_query_points


class NerfModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CONFIGS
        # Input dimensions for Positional Encoding : pos and dir
        self.L_pos = 10
        self.L_dir = 4
        pos_enc_features = 3 + 3 * 2 * self.L_pos
        dir_enc_features = 3 + 3 * 2 * self.L_dir
        in_features = pos_enc_features  # 63
        num_neurons = 256
        # early mlp layers = 5, with 256 neurons each

        # MLP
        self.early_mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        in_features = pos_enc_features + num_neurons  # 63 + 256
        # later mlp layers = 3, with 256 neurons each

        self.later_mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.sigma_layer = nn.Linear(num_neurons, num_neurons + 1)
        self.pre_final_layer = nn.Sequential(
            nn.Linear(dir_enc_features + num_neurons, num_neurons // 2),  # output 128 neuron layer
            nn.ReLU(),
        )
        self.final_layer = nn.Sequential(nn.Linear(num_neurons // 2, 3), nn.Sigmoid())  # rgb output

    def forward(self, rays_samples, view_dirs):
        # POSITIONAL ENCODING

        # rays_samples - 3D points
        rays_samples_encoded = [rays_samples]
        for l_pos in range(self.L_pos):
            rays_samples_encoded.append(torch.sin(2 ** l_pos * torch.pi * rays_samples))
            rays_samples_encoded.append(torch.cos(2 ** l_pos * torch.pi * rays_samples))

        rays_samples_encoded = torch.cat(rays_samples_encoded, dim=-1)

        # view_dirs - viewing directions of rays
        view_dirs = view_dirs / view_dirs.norm(p=2, dim=-1).unsqueeze(-1)
        view_dirs_encoded = [view_dirs]
        for l_dir in range(self.L_dir):
            view_dirs_encoded.append(torch.sin(2 ** l_dir * torch.pi * view_dirs))
            view_dirs_encoded.append(torch.cos(2 ** l_dir * torch.pi * view_dirs))

        view_dirs_encoded = torch.cat(view_dirs_encoded, dim=-1)

        # Use the network to predict colors (c_is) and volume densities (sigma_is) for
        # 3D points (xs) along rays given the viewing directions (ds) of the rays

        outputs = self.early_mlp(rays_samples_encoded)
        outputs = self.later_mlp(torch.cat([rays_samples_encoded, outputs], dim=-1))
        outputs = self.sigma_layer(outputs)
        sigma_is = torch.relu(outputs[:, 0])  # volume densities
        outputs = self.pre_final_layer(torch.cat([view_dirs_encoded, outputs[:, 1:]], dim=-1))
        c_is = self.final_layer(outputs)  # predicted colors

        return {"c_is": c_is, "sigma_is": sigma_is}


def main():
    # CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # seed
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    #  VOLUMETRIC RENDERING IN NERF - SEC:5.3
    coarse_mlp = NerfModel().to(device)
    fine_mlp = NerfModel().to(device)

    # Number of query points passed through the MLP at a time.
    chunk_size = 1024 * 32

    # Number of training rays per iteration. SEC:5.3
    batch_img_size = 64
    n_batch_pix = batch_img_size ** 2

    # INITIALIZE OPTIMIZER. SEC 5.3
    lr = 5e-4
    optimizer = optim.Adam(list(coarse_mlp.parameters()) + list(fine_mlp.parameters()), lr=lr)
    criterion = nn.MSELoss()
    # The learning rate decays exponentially. Section 5.3
    lrate_decay = 250
    decay_steps = lrate_decay * 1000
    decay_rate = 0.1

    # Load dataset.
    data_f = "66bdbc812bd0a196e194052f3f12cb2e.npz"
    data = np.load(data_f)

    # Set up initial ray origin (init_o) and ray directions (init_ds). These are the
    # same across samples, we just rotate them based on the orientation of the camera.
    # See Section 4.
    images = data["images"] / 255
    img_size = images.shape[1]
    xs = torch.arange(img_size) - (img_size / 2 - 0.5)
    ys = torch.arange(img_size) - (img_size / 2 - 0.5)
    (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
    focal = float(data["focal"])
    pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
    # We want the zs to be negative ones, so we divide everything by the focal length
    # (which is in pixel units).
    camera_coords = pixel_coords / focal
    init_ds = camera_coords.to(device)
    init_o = torch.Tensor(np.array([0, 0, float(data["camera_distance"])])).to(device)

    # Set up test view.
    test_idx = 150
    plt.imshow(images[test_idx])
    plt.show()
    test_img = torch.Tensor(images[test_idx]).to(device)
    poses = data["poses"]
    test_R = torch.Tensor(poses[test_idx, :3, :3]).to(device)
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)

    # VOLUME RENDERING HYPER-PARAMETERS - SEC:4
    t_n = 1.0  # Near bound.
    t_f = 4.0  # Far bound.
    num_coarse_sample = 64
    num_fine_sample = 128
    # Bins used to sample depths along a ray. SEC:4 Eq2
    """
    Here we basically  we use a stratified (arranged in layers) sampling approach 
    where we partition [tn, tf ] into N evenly-spaced bins 
    and then draw one sample uniformly at random from within each bin 
    """
    t_i_c_gap = (t_f - t_n) / num_coarse_sample
    t_i_c_bin_edges = (t_n + torch.arange(num_coarse_sample) * t_i_c_gap).to(device)

    train_idxs = np.arange(len(images)) != test_idx
    images = torch.Tensor(images[train_idxs])
    poses = torch.Tensor(poses[train_idxs])
    n_pix = img_size ** 2
    pixel_ps = torch.full((n_pix,), 1 / n_pix).to(device)
    psnrs = []
    iternums = []
    # See Section 5.3.
    num_iters = 300000
    display_every = 100
    coarse_mlp.train()
    fine_mlp.train()
    for i in range(num_iters):
        # Sample image and associated pose.
        target_img_idx = np.random.randint(images.shape[0])
        target_pose = poses[target_img_idx].to(device)
        R = target_pose[:3, :3]

        # Get rotated ray origins (os) and ray directions (ds). See Section 4.
        ds = torch.einsum("ij,hwj->hwi", R, init_ds)
        os = (R @ init_o).expand(ds.shape)

        # Sample a batch of rays.
        pix_idxs = pixel_ps.multinomial(n_batch_pix, False)
        pix_idx_rows = pix_idxs // img_size
        pix_idx_cols = pix_idxs % img_size
        ds_batch = ds[pix_idx_rows, pix_idx_cols].reshape(
            batch_img_size, batch_img_size, -1
        )
        os_batch = os[pix_idx_rows, pix_idx_cols].reshape(
            batch_img_size, batch_img_size, -1
        )

        # Run NeRF.
        (C_rs_c, C_rs_f) = run_one_iter_of_nerf(
            ds_batch,
            num_coarse_sample,
            t_i_c_bin_edges,
            t_i_c_gap,
            os_batch,
            chunk_size,
            coarse_mlp,
            num_fine_sample,
            t_f,
            fine_mlp,
        )
        target_img = images[target_img_idx].to(device)
        target_img_batch = target_img[pix_idx_rows, pix_idx_cols].reshape(C_rs_f.shape)
        # Calculate the mean squared error for both the coarse and fine MLP models and
        # update the weights. See Equation (6) in Section 5.3.
        loss = criterion(C_rs_c, target_img_batch) + criterion(C_rs_f, target_img_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Exponentially decay learning rate. See Section 5.3 and:
        # https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/.
        for g in optimizer.param_groups:
            g["lr"] = lr * decay_rate ** (i / decay_steps)

        if i % display_every == 0:
            coarse_mlp.eval()
            fine_mlp.eval()
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

            loss = criterion(C_rs_f, test_img)
            print(f"Loss: {loss.item()}")
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(C_rs_f.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

            coarse_mlp.train()
            fine_mlp.train()

    print("Completed Training!")
    print("Saving model...")
    torch.save(fine_mlp, "/")
    print("Done!")






