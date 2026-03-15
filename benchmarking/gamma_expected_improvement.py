import torch
from torch import Tensor
from typing import Optional, Union

from botorch.acquisition.analytic import (
    AnalyticAcquisitionFunction,
    _ei_helper,
    _scaled_improvement,
)
from botorch.models.model import Model
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition.input_constructors import acqf_input_constructor


class GammaExpectedImprovement(AnalyticAcquisitionFunction):
    r"""
    FigBO / GammaExpectedImprovement:
    Expected Improvement augmented with a global-information-gain term.

    Acquisition:
        EI(x) + lambda_n * Gamma(x)

    where:
        lambda_n = eta / n
        n = number of observed training points

    Gamma(x) is approximated by Monte Carlo integration over the search domain:
        Gamma(x) ≈ (1 / L) * sum_l k_{n,x}(z_l)^T (K_{n,x} + sigma^2 I)^(-1) k_{n,x}(z_l)

    Notes
    -----
    - This implementation supports q=1 analytic acquisition evaluation.
    - `bounds` may have shape (2, d) or (d, 2); it will be normalized to (2, d).
    - This class assumes access to a GP-like model exposing `train_inputs`
      and `covar_module`.
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        bounds: Tensor,
        eta: float = 1.0,
        num_mc_samples: int = 100,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        noise_variance: float = 1e-6,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(model=model, posterior_transform=posterior_transform)

        best_f = torch.as_tensor(best_f)
        bounds = torch.as_tensor(bounds, dtype=best_f.dtype, device=best_f.device)

        # Accept both (2, d) and (d, 2); normalize to (2, d)
        if bounds.ndim != 2:
            raise ValueError("`bounds` must be a 2D tensor-like object.")

        if bounds.shape[0] == 2:
            pass
        elif bounds.shape[1] == 2:
            bounds = bounds.transpose(0, 1)
        else:
            raise ValueError(
                f"`bounds` must have shape (2, d) or (d, 2), got {tuple(bounds.shape)}."
            )

        self.register_buffer("best_f", best_f)
        self.register_buffer("bounds", bounds)

        self.maximize = bool(maximize)
        self.eta = float(eta)
        self.num_mc_samples = int(num_mc_samples)
        self.noise_variance = float(noise_variance)
        self.jitter = float(jitter)

        if self.num_mc_samples <= 0:
            raise ValueError("`num_mc_samples` must be positive.")

        # training inputs from fitted GP model
        self.train_X = model.train_inputs[0]
        if self.train_X.ndim != 2:
            raise ValueError("Expected `model.train_inputs[0]` to have shape (n, d).")

        if self.train_X.shape[-1] != self.bounds.shape[-1]:
            raise ValueError(
                f"Dimension mismatch: train_X has d={self.train_X.shape[-1]}, "
                f"but bounds has d={self.bounds.shape[-1]}."
            )

    def _sample_points(self) -> Tensor:
        """Uniformly sample Monte Carlo points from the box domain."""
        low = self.bounds[0].to(dtype=self.train_X.dtype, device=self.train_X.device)
        high = self.bounds[1].to(dtype=self.train_X.dtype, device=self.train_X.device)

        if not torch.all(high > low):
            raise ValueError("Each upper bound must be strictly larger than the lower bound.")

        u = torch.rand(
            self.num_mc_samples,
            low.numel(),
            dtype=self.train_X.dtype,
            device=self.train_X.device,
        )
        return low + (high - low) * u

    def _compute_augmented_inverse(self, x_candidate: Tensor):
        """
        Build K_{n,x} + sigma^2 I and return its inverse.

        Args:
            x_candidate: Tensor of shape (d,)
        """
        x_candidate = x_candidate.view(1, -1)
        aug_X = torch.cat([self.train_X, x_candidate], dim=0)  # (n+1, d)

        K = self.model.covar_module(aug_X, aug_X).to_dense()
        n_aug = K.shape[0]

        eye = torch.eye(n_aug, dtype=K.dtype, device=K.device)
        K = K + (self.noise_variance + self.jitter) * eye

        return torch.linalg.inv(K), aug_X

    def _compute_gamma_for_candidate(self, x_candidate: Tensor, sampled_Z: Tensor) -> Tensor:
        """
        Compute Gamma(x_candidate) using Monte Carlo points sampled_Z.

        Args:
            x_candidate: Tensor of shape (d,)
            sampled_Z:   Tensor of shape (L, d)
        """
        K_inv, aug_X = self._compute_augmented_inverse(x_candidate)

        # k(z, [X_train; x_candidate]) -> shape (L, n+1)
        k_vec = self.model.covar_module(sampled_Z, aug_X).to_dense()

        # diag(k K^{-1} k^T), averaged over MC samples
        gamma_vals = (k_vec @ K_inv * k_vec).sum(dim=-1)  # (L,)
        return gamma_vals.mean()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""
        Evaluate FigBO acquisition on candidate set X.

        Args:
            X: A `(b1 x ... x bk) x 1 x d` tensor.

        Returns:
            A `(b1 x ... x bk)` tensor of acquisition values.
        """
        # Standard EI part, following BoTorch analytic EI style
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        ei_value = (sigma * _ei_helper(u)).squeeze(-1)  # (batch,)

        # Monte Carlo points for global term
        sampled_Z = self._sample_points()

        # Flatten batch dims for candidate-wise Gamma computation
        X_flat = X.squeeze(-2).reshape(-1, X.shape[-1])  # (B, d)

        gamma_list = []
        for x_candidate in X_flat:
            gamma_x = self._compute_gamma_for_candidate(x_candidate, sampled_Z)
            gamma_list.append(gamma_x)

        gamma_tensor = torch.stack(gamma_list, dim=0).view(ei_value.shape)

        # lambda_n = eta / n
        n_obs = max(int(self.train_X.shape[0]), 1)
        lambda_n = self.eta / float(n_obs)

        return ei_value + lambda_n * gamma_tensor


@acqf_input_constructor(GammaExpectedImprovement)
def construct_inputs_gamma_ei(
    model,
    training_data,
    bounds,
    eta=1.0,
    num_mc_samples=100,
    objective_thresholds=None,
    posterior_transform=None,
    maximize=True,
    noise_variance=1e-6,
    jitter=1e-6,
):
    """
    Input constructor for GammaExpectedImprovement.
    """
    Y = training_data.Y

    if maximize:
        best_f = Y.max()
    else:
        best_f = Y.min()

    return {
        "model": model,
        "best_f": best_f,
        "bounds": bounds,
        "eta": eta,
        "num_mc_samples": num_mc_samples,
        "posterior_transform": posterior_transform,
        "maximize": maximize,
        "noise_variance": noise_variance,
        "jitter": jitter,
    }