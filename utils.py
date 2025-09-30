import torch
import multipers as mp
from multipers.filtrations import Cubical
from multipers.torch.diff_grids import get_grid, evaluate_mod_in_grid
import numpy as np
import powerbox as pbox
from gudhi import PeriodicCubicalComplex
from tqdm.notebook import tqdm

def create_grid_from_data(filtration_values, strategy="regular_closest", resolution=50):
    """
    Creates a grid based on filtration values, optimized for PyTorch compatibility.
    This function generates a grid for evaluating bifiltration data, supporting different
    strategies (e.g., "regular_closest" or "exact") and resolutions.
    
    WARNING: Custom grids break gradient computation. Use grid=None in compute_mma_descriptor
    for differentiable operations.
    
    Args:
        filtration_values (list[torch.Tensor]): A list of 1D PyTorch tensors,
            each representing filtration values for a parameter.
        strategy (str, optional): The grid generation strategy. Defaults to "regular_closest".
        resolution (int or list[int], optional): The resolution for grid creation.
            If an integer, the same resolution is used for all parameters. Defaults to 50.
    
    Returns:
        tuple: A grid optimized for the input data, compatible with PyTorch operations.
    """
    grid_function = get_grid(strategy)
    if strategy == "exact":
        grid = grid_function(filtration_values)
    else:
        if isinstance(resolution, int):
            resolution = [resolution] * len(filtration_values)
        grid = grid_function(filtration_values, resolution)
    return grid

def compute_mma_descriptor(field, derivative, degree=1, grid=None):
    """
    Computes the MMA (Multiparameter Module Approximation) descriptor for a scalar field
    and its derivative on a 2D grid.
    
    This function processes a scalar field and its derivative, constructs a bifiltration,
    and computes the topological descriptor using cubical filtrations. The output is
    differentiable if inputs are PyTorch tensors with gradients enabled.
    
    Parameters:
        field (torch.Tensor or np.ndarray): A 2D tensor/array of shape (n, m)
            representing the scalar field.
        derivative (torch.Tensor or np.ndarray): A 2D tensor/array of shape (n, m)
            representing the derivative of the field.
        degree (int, optional): The degree of the homology group to evaluate. Defaults to 1.
        grid (tuple, optional): A precomputed grid for evaluation. If None, a grid is
            automatically generated. Defaults to None.
            WARNING: Custom grids break gradient computation. Use grid=None for differentiable operations.
    
    Returns:
        tuple:
            - result (torch.Tensor): The evaluated MMA descriptor for the specified degree.
            - mma (object): The module approximation object for further analysis.
    
    Notes:
        - If NumPy arrays are passed as inputs, they are automatically converted to PyTorch tensors.
        - The function preserves differentiability when PyTorch tensors with gradients are used ONLY with grid=None.
        - Custom grids break the gradient computation due to limitations in the multipers library.
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(field, torch.Tensor):
        field = torch.from_numpy(field)
    if not isinstance(derivative, torch.Tensor):
        derivative = torch.from_numpy(derivative)
    
    # Construct bifiltration (stack field and derivative along the last dimension)
    bifiltration = torch.stack([field, derivative], dim=-1)
    temp = Cubical(bifiltration)
    mma = mp.module_approximation(temp)
    
    # Generate grid if not provided
    if grid is None:
        filtration_values = [field.flatten(), derivative.flatten()]
        grid = create_grid_from_data(filtration_values, strategy='exact')
    
    result = evaluate_mod_in_grid(mma, grid)
    
    if degree >= len(result):
        print(f"Warning: grado {degree} non disponibile, gradi disponibili: {list(range(len(result)))}")
        return None, mma
    
    # Evaluate the module approximation on the grid
    return result[degree], mma

def simulate_grfs(thet, n_sims, seed = None, physical = False, N = 32):
    a, b = thet
    if(seed != None) : np.random.seed(seed)
    seeds = np.random.randint(1e7, size = (n_sims))
    lis = []
    for i in range(n_sims):
        pb = pbox.PowerBox(
            N=N,                     # Number of grid-points in the box
            dim=2,                     # 2D box
            pk = lambda k: a*k**(-b), # The power-spectrum
            boxlength = N,           # Size of the box (sets the units of k in pk)
            vol_normalised_power=True, 
            ensure_physical = physical,
            seed = seeds[i]
        )
        lis.append(pb.delta_x())
    return np.array(lis)

def compute_gradient(field, x, y):
    gradient = np.gradient(field, x, y)
    return gradient[0] + gradient[1]


# --- 2) batch of fields -> list of diagrams ---
def batch_diagrams(fields, dims=(0, 1), periodic=(True, True), n_max=None):
    """
    fields: ndarray of shape (n_sims, N, N) or iterable of (N, N)
    returns: list of dicts {dim: intervals}
    """
    if isinstance(fields, np.ndarray) and fields.ndim == 3:
        it = fields[:n_max] if n_max else fields
    else:
        it = fields  # e.g. a list
    out = []
    for f in tqdm(it):
        out.append(diagrams_from_field(f, dims=dims, periodic=periodic))
    return out