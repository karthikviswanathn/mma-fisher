import powerbox as pbox
import numpy as np
# from gudhi import PeriodicCubicalComplex
# from gudhi.representations import PersistenceImage, Landscape
from tqdm.notebook import tqdm


def _finite_intervals(diag, max_death=None):
    diag = np.asarray(diag, dtype=float)
    if diag.size == 0:
        return diag
    if max_death is None:
        # heuristic: cap at 99% finite death or max finite death
        finite = np.isfinite(diag[:,1])
        max_death = np.percentile(diag[finite,1], 99) if finite.any() else 1.0
    diag = diag.copy()
    diag[:,1] = np.where(np.isfinite(diag[:,1]), diag[:,1], max_death)
    return diag

def tqdm_10(n_sim, desc="Progress"):
    return tqdm(
        range(n_sim),
        total=n_sim,
        miniters=max(1, n_sim // 10),  # updates every ~10%
        desc=desc
    )
    
##### Many choices in this file assume that theta is 2 dimensional and tailored to make
##### things work. More customization is needed in case we want to extend this to 
##### other problems.

def _numpy(inp):
    return inp.numpy() if isinstance(inp, torch.Tensor) else inp

class GRF:
    def __init__(self, physical=True, N=32, theta=[1., 2.], seed=None):
        self.physical = physical
        self.N = N
        self.a = _numpy(theta[0])
        self.b = _numpy(theta[1])
        self.seed = seed if seed is not None else np.random.randint(1e7)
        self.data = self.simulate_data()
        self.diagrams = self.find_diagrams(self.data)
        self.persistence_images = self.summarize_diagrams(self.diagrams)

    def simulate_data(self):
        pb = pbox.PowerBox(
                N=self.N,                     # Number of grid-points in the box
                dim=2,                     # 2D box
                pk = lambda k: self.a*k**(-self.b), # The power-spectrum
                boxlength = self.N,           # Size of the box (sets the units of k in pk)
                vol_normalised_power=True, 
                ensure_physical = self.physical,
                seed = self.seed
            )
        return pb.delta_x()

    def find_diagrams(self, field_2d, dims=(0, 1), periodic=(True, True),
                            homology_coeff_field=2, min_persistence=0.0):
        # field_2d: ndarray of shape (N, N) with filtration values
        pcc = PeriodicCubicalComplex(
            top_dimensional_cells=field_2d,  # keep it light
            periodic_dimensions=list(periodic)
        )
        pcc.persistence(homology_coeff_field=homology_coeff_field,
                        min_persistence=min_persistence)
        return {d: pcc.persistence_intervals_in_dimension(d) for d in dims}
    
    
    def summarize_diagrams(self, diags):
        images = []
        for d, diag in diags.items():
            images.append(self.persistence_image(diag))
        return np.stack(images, axis = 0)

    # --- Persistence Images ---
    def persistence_image(self, diag, resolution=(32, 32), bandwidth=0.05,
                          weight_power=1.0):
        """
        Returns: flat PI vector (shape: resolution[0]*resolution[1]) and image (2D array).
        """
        diag = _finite_intervals(diag)
        if diag.size == 0:
            vec = np.zeros(resolution[0]*resolution[1], dtype=float)
            return vec, vec.reshape(resolution)
        a_np = self.a.numpy() if isinstance(self.a, torch.Tensor) else self.a
        PI = PersistenceImage(
            bandwidth=bandwidth,
            resolution=resolution,
            im_range = np.array([-2, 2, 0, 2])*a_np# Look into this setting, hardcoding this for now
            # im_range=im_range  # e.g., [bmin, bmax, dmin, dmax]
        )
        
        # fit_transform expects a list of diagrams
        vec = PI.fit_transform([diag])[0]   # shape = (resolution[0]*resolution[1],)
        # print(PI.im_range)
        return vec.reshape(resolution)

# class FisherSimulator:
    # def __init__(self, n_cov, n_der, save_path, load_path, simulator = GRF, theta_fid = [1,2], delta_theta = [0.1, 0.1]):
        
    #     self.delta_theta = torch.tensor(delta_theta, dtype=torch.float32)
    #     self.theta_fid = theta_fid
    #     self.simulator = simulator
    #     self.load_simulations(save_path = save_path, 
    #                           load_path = load_path, 
    #                           n_cov = n_cov, 
    #                           n_der = n_der) 
    #     self.model = SimpleCNNWithDropout(n_outputs=len(theta_fid), hidden=16, dropout_p = 0.4)
        
    # def load_simulations(self, save_path, load_path, n_cov, n_der):
    #     if load_path is not None and os.path.exists(load_path):
    #         print(f"Loading simulations from {load_path}")
    #         data = np.load(load_path, allow_pickle=True)
    #         self.cov_simulations = data["cov_simulations"].tolist()
    #         self.deriv_simulations = data["deriv_simulations"].tolist()
    #     else:
    #         print("Generating simulations...")
    #         theta_fid = self.theta_fid
    #         delta_theta = self.delta_theta
    #         self.cov_simulations = [self.simulator(theta = theta_fid) for _ in tqdm_10(n_cov)]
    #         theta_derivatives = [[theta_fid[0] - 0.5 * delta_theta[0], theta_fid[1]],
    #                          [theta_fid[0] + 0.5 * delta_theta[0], theta_fid[1]],
    #                          [theta_fid[0], theta_fid[1] - 0.5 * delta_theta[1]],
    #                          [theta_fid[0], theta_fid[1] + 0.5 * delta_theta[1]]]

    #         self.deriv_simulations = []
    #         seeds = np.random.randint(1, int(1e6) + 1, size=(len(theta_fid), n_der))
    #         seeds = np.repeat(seeds, repeats=2, axis=0)
    #         self.seeds = seeds
    #         # Do seed fixing here
    #         for theta_index, theta in enumerate(theta_derivatives):
    #             self.deriv_simulations.append([self.simulator(theta = theta, seed = seeds[theta_index, idx]) for idx in tqdm_10(n_der, desc=f"Deriv sims {(theta_index)}")])

            
    #         if save_path is not None:
    #             print(f"Saving simulations to {save_path}")
    #             np.savez_compressed(
    #                 save_path,
    #                 cov_simulations=np.array(self.cov_simulations, dtype=object),
    #                 deriv_simulations=np.array(self.deriv_simulations, dtype=object),
    #             )
    # def stack_images(self, simulations):
    #     arr = np.stack([sim.persistence_images for sim in simulations], axis=0)
    #     return torch.from_numpy(arr).float()
    
    # def compress_images(self, simulations):
    #     images = self.stack_images(simulations)
    #     return self.model(images)
        
    # def forward_pass(self):
    #     cov_summary = self.compress_images(self.cov_simulations)
    #     deriv_summary = [self.compress_images(sim) for sim in self.deriv_simulations] # Summaries used to calculate derivatives
    #     deriv_summary = torch.stack(deriv_summary) # (4, B, 2)
    #     Sigma = torch.cov(cov_summary.T, correction=1)
    #     derivatives_mean = (deriv_summary[1::2] - deriv_summary[::2]).mean(axis = 1)/self.delta_theta.unsqueeze(0)
    #     # print("Derivatives.shape", (deriv_summary[1::2] - deriv_summary[::2]).mean(axis = 1).shape)
    #     # print("Theta.shape = ", self.delta_theta.unsqueeze(-1).shape)
    #     Sigma_inv = torch.linalg.inv(Sigma)
    #     F = derivatives_mean.T @ Sigma_inv @ derivatives_mean
    #     return {'logdetF': -torch.slogdet(F)[1], 'F': F}
    
    # def train_model(
    #     self,
    #     epochs=50,
    #     lr=1e-3,
    #     weight_decay=0.0,
    #     clip_grad_norm=None,
    #     log_every=20,
    # ):
    #     # device setup
    #     self.model.train()

    #     opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    #     history = {"loss": [], 'FM': []}
    #     for ep in range(1, epochs + 1):
    #         opt.zero_grad()
    #         output = self.forward_pass()
    #         loss = output['logdetF']  # should return scalar tensor (your code returns -slogdet(F))
    #         # ensure it's on same device & dtype
    #         loss = loss.float()

    #         loss.backward()
    #         if clip_grad_norm is not None:
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
    #         opt.step()

    #         history["loss"].append(float(loss.detach().cpu()))
    #         history["FM"].append(output['F'].detach().numpy())
    #         if log_every and (ep % log_every == 0 or ep == 1 or ep == epochs):
    #             print(f"[{ep:04d}/{epochs}] loss={history['loss'][-1]:.2f}  (−logdetF)")

    #     return history
    
