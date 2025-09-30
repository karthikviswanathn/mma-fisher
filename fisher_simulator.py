import torch
import torch.nn as nn
import torch.nn.functional as F
from grf_simulator import GRF
import os
import numpy as np
import powerbox as pbox
from tqdm.notebook import tqdm


def fisher(θ, k, N):
    A, B = θ
    k = k[0:N//2, 0:N].flatten()
    Pk = A*k**(-B)

    Cinv = np.diag(1. / Pk)
    C_A =  np.diag(k ** -B)
    C_B =  np.diag(- Pk * np.log(k))

    F_AA = 0.5 * np.trace((C_A @ Cinv @ C_A @ Cinv))
    F_AB = 0.5 * np.trace((C_A @ Cinv @ C_B @ Cinv))
    F_BA = 0.5 * np.trace((C_B @ Cinv @ C_A @ Cinv))
    F_BB = 0.5 * np.trace((C_B @ Cinv @ C_B @ Cinv))

    return np.array([[F_AA, F_AB], [F_BA, F_BB]])


class SimpleCNNWithBNDropout(nn.Module):
    """
    Basic CNN for 32x32 images with BatchNorm + Dropout regularization.
    Architecture:
      Conv(in->8, 3x3) -> BatchNorm -> ReLU -> Dropout -> MaxPool(2)
      Conv(8->16, 3x3) -> BatchNorm -> ReLU -> Dropout -> MaxPool(2)
      Flatten -> Linear(16*8*8 -> hidden) -> ReLU -> Dropout -> Linear(hidden -> n_outputs)
    """
    def __init__(self, n_outputs, hidden, in_channels=2, dropout_p_conv=0.2, dropout_p_fc=0.5):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.dropout1 = nn.Dropout2d(p=dropout_p_conv)

        # Second conv block
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.dropout2 = nn.Dropout2d(p=dropout_p_conv)

        # Fully connected layers
        self.fc1 = nn.Linear(4 * 8 * 8, hidden)
        self.dropout3 = nn.Dropout(p=dropout_p_fc)
        self.fc2 = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)   # -> 16x16

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)   # -> 8x8

        # Fully connected
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        return self.fc2(x)


class SimpleCNNWithDropout(nn.Module):
    """
    Basic CNN for 32x32 images with dropout regularization.
    Architecture:
      Conv(1->8, 3x3) -> ReLU -> Dropout -> MaxPool(2)
      Conv(8->16, 3x3) -> ReLU -> Dropout -> MaxPool(2)
      Flatten -> Linear(16*8*8 -> hidden) -> ReLU -> Dropout -> Linear(hidden -> n_outputs)
    """
    def __init__(self, n_outputs, hidden, in_channels=2, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(p=dropout_p)  # for conv layers

        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_p)

        self.fc1 = nn.Linear(4 * 8 * 8, hidden)
        self.dropout3 = nn.Dropout(p=dropout_p)    # for fully connected layers

        self.fc2 = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        # x: (batch, in_channels, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)  # -> 16x16

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)  # -> 8x8

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        return self.fc2(x)
            

class SimpleCNN(nn.Module):
    """
    Basic CNN for 32x32 single-channel images.
    Architecture:
      Conv(1->8, 3x3) -> ReLU -> MaxPool(2)
      Conv(8->16, 3x3) -> ReLU -> MaxPool(2)
      Flatten -> Linear(16*8*8 -> hidden) -> ReLU -> Linear(hidden -> n_outputs)
    """
    def __init__(self, n_outputs, hidden=64, in_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        # x: (batch, 1, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 16x16
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 8x8
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class FisherSimulator:
    def __init__(self, n_cov, n_der, save_path, load_path, simulator = GRF, theta_fid = [1,2], delta_theta = [0.1, 0.1], val_split=0.2):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delta_theta = torch.tensor(delta_theta, dtype=torch.float32, device=self.device)
        self.theta_fid = theta_fid
        self.simulator = simulator
        self.val_split = val_split
        self.load_simulations(save_path = save_path, 
                              load_path = load_path, 
                              n_cov = n_cov, 
                              n_der = n_der) 
        self.split_train_val()
        # Select device
        
        # Initialize model on that device
        self.model = SimpleCNNWithBNDropout(
            n_outputs=len(theta_fid), hidden=8
        ).to(self.device)
        
    def load_simulations(self, save_path, load_path, n_cov, n_der):
        if load_path is not None and os.path.exists(load_path):
            print(f"Loading simulations from {load_path}")
            data = np.load(load_path, allow_pickle=True)
            self.cov_simulations = data["cov_simulations"].tolist()
            self.deriv_simulations = data["deriv_simulations"].tolist()
        else:
            print("Generating simulations...")
            theta_fid = self.theta_fid
            delta_theta = self.delta_theta
            self.cov_simulations = [self.simulator(theta = theta_fid) for _ in tqdm(n_cov)]
            theta_derivatives = [[theta_fid[0] - 0.5 * delta_theta[0], theta_fid[1]],
                             [theta_fid[0] + 0.5 * delta_theta[0], theta_fid[1]],
                             [theta_fid[0], theta_fid[1] - 0.5 * delta_theta[1]],
                             [theta_fid[0], theta_fid[1] + 0.5 * delta_theta[1]]]

            self.deriv_simulations = []
            seeds = np.random.randint(1, int(1e6) + 1, size=(len(theta_fid), n_der))
            seeds = np.repeat(seeds, repeats=2, axis=0)
            self.seeds = seeds
            # Do seed fixing here
            for theta_index, theta in enumerate(theta_derivatives):
                self.deriv_simulations.append([self.simulator(theta = theta, seed = seeds[theta_index, idx]) for idx in tqdm_10(n_der, desc=f"Deriv sims {(theta_index)}")])

            
            if save_path is not None:
                print(f"Saving simulations to {save_path}")
                np.savez_compressed(
                    save_path,
                    cov_simulations=np.array(self.cov_simulations, dtype=object),
                    deriv_simulations=np.array(self.deriv_simulations, dtype=object),
                )
    
    def split_train_val(self):
        """Split the loaded simulations into train and validation sets"""
        n_cov = len(self.cov_simulations)
        n_der = len(self.deriv_simulations[0])
        
        n_cov_val = int(n_cov * self.val_split)
        n_der_val = int(n_der * self.val_split)
        
        # Split covariance simulations
        self.cov_simulations_train = self.cov_simulations[:-n_cov_val] if n_cov_val > 0 else self.cov_simulations
        self.cov_simulations_val = self.cov_simulations[-n_cov_val:] if n_cov_val > 0 else None
        
        # Split derivative simulations
        self.deriv_simulations_train = []
        self.deriv_simulations_val = []
        
        for deriv_sim in self.deriv_simulations:
            if n_der_val > 0:
                self.deriv_simulations_train.append(deriv_sim[:-n_der_val])
                self.deriv_simulations_val.append(deriv_sim[-n_der_val:])
            else:
                self.deriv_simulations_train.append(deriv_sim)
                self.deriv_simulations_val = None
                
    def stack_images(self, simulations):
        arr = np.stack([sim.persistence_images for sim in simulations], axis=0)
        return torch.from_numpy(arr).float().to(self.device)
    
    def compress_images(self, simulations):
        images = self.stack_images(simulations)
        return self.model(images)
        

    def forward_pass(self, use_val=False):
        if use_val:
            cov_summary = self.compress_images(self.cov_simulations_val)
            deriv_summary = [self.compress_images(sim) for sim in self.deriv_simulations_val]
        else:
            cov_summary = self.compress_images(self.cov_simulations_train)
            deriv_summary = [self.compress_images(sim) for sim in self.deriv_simulations_train]
        
        # deriv_summary is a list of 4 tensors: [θ₁⁻, θ₁⁺, θ₂⁻, θ₂⁺]
        # Each tensor has shape (n_simulations, n_outputs)
        deriv_summary = torch.stack(deriv_summary)  # Shape: (4, n_simulations, n_outputs)
        
        # Compute covariance matrix from fiducial simulations
        Sigma = torch.cov(cov_summary.T, correction=1)  # Shape: (n_outputs, n_outputs)
        
        # Compute derivatives correctly
        # For parameter 1: (θ₁⁺ - θ₁⁻) / Δθ₁
        deriv_theta1 = (deriv_summary[1] - deriv_summary[0]) / self.delta_theta[0]  # Shape: (n_simulations, n_outputs)
        # For parameter 2: (θ₂⁺ - θ₂⁻) / Δθ₂
        deriv_theta2 = (deriv_summary[3] - deriv_summary[2]) / self.delta_theta[1]  # Shape: (n_simulations, n_outputs)
        
        # Stack and compute mean derivatives
        derivatives = torch.stack([deriv_theta1, deriv_theta2], dim=0)  # Shape: (2, n_simulations, n_outputs)
        derivatives_mean = derivatives.mean(dim=1)  # Shape: (2, n_outputs)

        # Compute Fisher matrix with improved regularization
        # Use trace-based regularization to prevent ill-conditioning
        reg_strength = 1e-3  # Tunable: increase if Fisher still exceeds theoretical
        I = torch.eye(Sigma.shape[0], device=Sigma.device, dtype=Sigma.dtype)
        Sigma_reg = Sigma + reg_strength * torch.trace(Sigma) * I / Sigma.shape[0]
        Sigma_inv = torch.linalg.inv(Sigma_reg)
        F = derivatives_mean @ Sigma_inv @ derivatives_mean.T  # Shape: (2, 2)

        # logdetF (note: slogdet is safer)
        sign, logabsdetF = torch.slogdet(F + 1e-9 * torch.eye(F.shape[0], device=F.device, dtype=F.dtype))
        # In case of pathological sign, still use logabsdet
        neg_logdetF = - sign * logabsdetF

        return {
            'neg_logdetF': neg_logdetF,
            'F': F,
            'C': Sigma,
            'C_inv': Sigma_inv,
        }
        
        # return {'logdetF': -torch.slogdet(F)[1], 'F': F}

    @staticmethod
    def _cov_regulariser(C, lambda_reg=10.0, alpha=0.01):
        """
        Compute IMNN covariance regulariser.
        Args:
            C: (n_out, n_out) covariance of summaries
            lambda_reg: λ in docs
            alpha: α in docs (shapes decay of regularisation)
        Returns:
            lam2: Λ2
            r: r(Λ2)
        """
        I = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
        # Frobenius norms
        diff = C - I
        invC = torch.linalg.inv(C + 1e-6 * I)
        diff_inv = invC - I
        lam2 = torch.linalg.norm(diff, ord='fro') + torch.linalg.norm(diff_inv, ord='fro')

        # r(Λ2) = λ Λ2 / (Λ2 - exp(-α Λ2))
        # add a tiny epsilon for numerical safety when Λ2 is very small
        eps = torch.as_tensor(1e-12, device=C.device, dtype=C.dtype)
        r = lambda_reg * lam2 / (torch.clamp(lam2 - torch.exp(-alpha * lam2), min=eps))
        return lam2, r

    def compute_theoretical_fisher(self, N=32):
        """Compute theoretical Fisher Information Matrix for GRF with power spectrum pk = A*k^(-B)"""
        A, B = self.theta_fid[0], self.theta_fid[1]
        theta = [A, B]
        pb = pbox.PowerBox(
            N = N,
            dim = 2,
            pk = lambda k: A*k**(-B),
            boxlength = N,
            vol_normalised_power=True,
            ensure_physical = False,
            seed = 42
        )
        theoretical_fisher = fisher(theta, pb.k(), N)
        return theoretical_fisher

    def train_model(
        self,
        epochs=50,
        lr=1e-3,
        weight_decay=0.0,
        clip_grad_norm=None,
        log_every=20,
    ):
        # device setup
        self.model.train()

        # Compute and print theoretical Fisher matrix
        theoretical_fisher = self.compute_theoretical_fisher()
        if theoretical_fisher is not None:
            logdet_theoretical = np.linalg.slogdet(theoretical_fisher)[1]
            print()
            print(f"Theoretical: logdetF={logdet_theoretical:.2f}")
            print(f"Theoretical Fisher Matrix: {[[round(x, 2) for x in row] for row in theoretical_fisher.tolist()]}")
            print()

        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        history = {"loss": [], 'FM': [], "val_loss": [], 'val_FM': []}
        for ep in range(1, epochs + 1):
            # Training step
            self.model.train()
            opt.zero_grad()
            output = self.forward_pass()
            loss = output['neg_logdetF']  # should return scalar tensor (your code returns -slogdet(F))
            # ensure it's on same device & dtype
            # loss = loss.float()

            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
            opt.step()

            history["loss"].append(float(loss.detach().cpu()))
            history["FM"].append(output['F'].detach().cpu().numpy())
            
            # Validation step
            if self.cov_simulations_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.forward_pass(use_val=True)
                    val_loss = val_output['neg_logdetF'].float()
                    history["val_loss"].append(float(val_loss.detach().cpu()))
                    history["val_FM"].append(val_output['F'].detach().cpu().numpy())
            
            if log_every and (ep % log_every == 0 or ep == 1 or ep == epochs):
                if self.cov_simulations_val is not None:
                    print(f"[{ep:04d}/{epochs}], loss={history['loss'][-1]:.2f}, val_loss={history['val_loss'][-1]:.2f}") 
                    print(f"Fisher Matrix: {[[round(x, 2) for x in row] for row in history['FM'][-1].tolist()]}")
                    print(f"val Fisher Matrix: {[[round(x, 2) for x in row] for row in history['val_FM'][-1].tolist()]}")
                    print()
                else:
                    print(f"[{ep:04d}/{epochs}] loss={history['loss'][-1]:.2f}  (−logdetF)")
                    print(f"Fisher Matrix: {[[round(x, 2) for x in row] for row in history['FM'][-1].tolist()]}")
                    print()
        
        return history

    # def train_model_reg(
    #         self,
    #         epochs=50,
    #         lr=1e-3,
    #         weight_decay=0.0,
    #         clip_grad_norm=None,
    #         log_every=20,
    #         lambda_reg=1.0,   # λ
    #         alpha=None,       # α (set either alpha or epsilon)
    #         epsilon=None,     # ε -> used to set α if provided
    #     ):
    #     self.model.train()
    #     opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    #     # choose alpha from epsilon if given
    #     if alpha is None:
    #         if epsilon is None:
    #             alpha = 1.0  # a reasonable default
    #         else:
    #             # map ε ("closeness") to α; simple monotone mapping
    #             alpha = float(1.0 / max(epsilon, 1e-6))

    #     history = {"loss": [], "reg": [], "lam2": [], 'FM': [], "val_loss": [], "val_reg": [], "val_lam2": [], 'val_FM': []}

    #     for ep in tqdm(range(1, epochs + 1)):
    #         self.model.train()
    #         opt.zero_grad()

    #         out = self.forward_pass()
    #         neg_logdetF, C = out['neg_logdetF'], out['C']

    #         # IMNN covariance regulariser
    #         lam2, r = self._cov_regulariser(C, lambda_reg=lambda_reg, alpha=alpha)
    #         loss = neg_logdetF + r * lam2

    #         loss = loss.float()
    #         loss.backward()
    #         if clip_grad_norm is not None:
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
    #         opt.step()

    #         history["loss"].append(float(loss.detach().cpu()))
    #         history["reg"].append(float((r * lam2).detach().cpu()))
    #         history["lam2"].append(float(lam2.detach().cpu()))
    #         history["FM"].append(out['F'].detach().cpu().numpy())

    #         # Validation
    #         if self.cov_simulations_val is not None:
    #             self.model.eval()
    #             with torch.no_grad():
    #                 vout = self.forward_pass(use_val=True)
    #                 v_neg_logdetF, vC = vout['neg_logdetF'], vout['C']
    #                 v_lam2, v_r = self._cov_regulariser(vC, lambda_reg=lambda_reg, alpha=alpha)
    #                 v_loss = v_neg_logdetF + v_r * v_lam2

    #                 history["val_loss"].append(float(v_loss.detach().cpu()))
    #                 history["val_reg"].append(float((v_r * v_lam2).detach().cpu()))
    #                 history["val_lam2"].append(float(v_lam2.detach().cpu()))
    #                 history["val_FM"].append(vout['F'].detach().cpu().numpy())

    #         if log_every and (ep % log_every == 0 or ep == 1 or ep == epochs):
    #             if self.cov_simulations_val is not None and history["val_loss"]:
    #                 print(f"[{ep:04d}/{epochs}] loss={history['loss'][-1]:.3f}  reg={history['reg'][-1]:.3f}  Λ2={history['lam2'][-1]:.3f}  |  "
    #                       f"val_loss={history['val_loss'][-1]:.3f}  val_reg={history['val_reg'][-1]:.3f}  val_Λ2={history['val_lam2'][-1]:.3f}")
    #             else:
    #                 print(f"[{ep:04d}/{epochs}] loss={history['loss'][-1]:.3f}  reg={history['reg'][-1]:.3f}  Λ2={history['lam2'][-1]:.3f}")

    #     return history
