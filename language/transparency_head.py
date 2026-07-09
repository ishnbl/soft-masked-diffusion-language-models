#
# Copyright 2026- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inv_param(init) -> torch.Tensor:
    """
    Given a desired initial value `init` for a parameter that will be transformed.
    This will ensure that the parameter is always positive.
    """
    return nn.Parameter(torch.log(torch.expm1(torch.tensor(init, dtype=torch.float32))))


def frechet_mean_sphere(vhat, weights, n_iter, eps):
    """
    Weighted Frechet (Karcher) mean of points on the unit hypersphere S^{d-1}.

    vhat:    (..., k, D) unit vectors (already L2-normalised over D)
    weights: (..., k)    non-negative weights that sum to 1 over the k axis
    Returns: (..., D)    unit vector
    """
    # Initialise at the highest-weight point (top-1 token).
    mu = F.normalize(vhat[..., 0, :], dim=-1, eps=eps)  # (..., D)
    w = weights.unsqueeze(-1)  # (..., k, 1)
    for _ in range(n_iter):
        # cos of the angle between mu and each v_i
        cos = (mu.unsqueeze(-2) * vhat).sum(-1).clamp(-1 + eps, 1 - eps)  # (..., k)
        omega = torch.acos(cos)  # (..., k)
        sin_omega = torch.sin(omega)
        # log map at mu: ratio = omega / sin(omega) -> 1 as omega -> 0
        ratio = torch.where(
            sin_omega > eps, omega / sin_omega.clamp_min(eps), torch.ones_like(omega)
        )
        tang = ratio.unsqueeze(-1) * (
            vhat - cos.unsqueeze(-1) * mu.unsqueeze(-2)
        )  # (..., k, D)
        tau = (w * tang).sum(-2)  # (..., D)
        tau_norm = tau.norm(dim=-1, keepdim=True)  # (..., 1)
        direction = tau / tau_norm.clamp_min(eps)
        # exp map back onto the sphere; as tau_norm -> 0 this leaves mu unchanged
        mu = torch.cos(tau_norm) * mu + torch.sin(tau_norm) * direction
        mu = F.normalize(mu, dim=-1, eps=eps)
    return mu


def slerp_sm_feedback(
    input_ids,
    logits,
    embedding_matrix,
    mask_token_id,
    lambda_tensor,
    top_k,
    n_iter=3,
    eps=1e-6,
    stats=None,
):
    """
    Soft-mask feedback in embedding space.

    For each masked position we SLERP between the (normalised) mask-token
    embedding and the Frechet mean of the top-k predicted token embeddings on
    the unit hypersphere. Unmasked positions keep their own token embedding.

    input_ids:        (B, T)     current (partially masked) token ids
    logits:           (B, T, V)  feedback logits from the previous pass
    embedding_matrix: (V, D)     token embedding table E
    lambda_tensor:    (B, T, 1)  SLERP weight, 0 on non-mask positions
    stats:            optional dict; if given, gets "slerp_angle_mean" (the mean
                      SLERP angle over masked positions) for live logging
    Returns:          (B, T, D)  soft input embeddings
    """
    # changed

    compute_dtype = torch.float32
    E = embedding_matrix  # (V, D) — index only, no full cast

    mask_pos = input_ids == mask_token_id  # (B, T)

    if not mask_pos.any():
        return E[input_ids].to(embedding_matrix.dtype)

    out = E[input_ids].to(
        compute_dtype
    )  # only allocated when there are masked positions

    # --- Operate only on the M masked positions to avoid wasted topk over (B,T,V). ---
    masked_logits = logits[mask_pos]  # (M, V)
    lam = lambda_tensor.squeeze(-1)[mask_pos].to(compute_dtype).unsqueeze(-1)  # (M, 1)

    # Top-k tokens and renormalised weights pi (== softmax over the top-k logits).
    topk_logits, topk_indices = torch.topk(masked_logits, k=top_k, dim=-1)  # (M, k)
    pi = torch.softmax(topk_logits.to(compute_dtype), dim=-1)  # (M, k)

    # Unit embeddings of the top-k tokens and Frechet mean mu*.
    # AFTER
    vhat = F.normalize(E[topk_indices].to(compute_dtype), dim=-1, eps=eps)  # (M, k, D)
    mu = frechet_mean_sphere(vhat, pi, n_iter, eps)  # (M, D)

    # Normalised mask embedding m_hat. Keep the original mask-token norm so we
    # can rescale the unit-sphere SLERP result back to the embedding scale the
    # backbone was trained on (otherwise norm-1 inputs are out of distribution).
    # AFTER
    mask_emb = E[mask_token_id].to(compute_dtype)  # (D,)
    mask_norm = mask_emb.norm()  # scalar
    mhat = F.normalize(mask_emb, dim=-1, eps=eps).expand_as(mu)  # (M, D)

    # SLERP(m_hat, mu*, lambda).
    cos = (mhat * mu).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)  # (M, 1)
    omega = torch.acos(cos)  # (M, 1)
    sin_omega = torch.sin(omega).clamp_min(eps)
    coeff_m = torch.sin((1 - lam) * omega) / sin_omega
    coeff_mu = torch.sin(lam * omega) / sin_omega
    slerp = coeff_m * mhat + coeff_mu * mu  # (M, D)
    # When mask and mean are ~identical the SLERP is undefined -> return m_hat.
    slerp = torch.where(omega < eps, mhat, slerp)
    # The SLERP lives on the unit hypersphere (norm 1). Rescale back to the
    # original mask-token norm so the embeddings stay in the backbone's
    # training distribution rather than collapsing onto the unit sphere.
    slerp = slerp * mask_norm  # (M, D)

    # Scatter slerp results back into the full (B,T,D) output tensor.
    # changed
    # out[mask_pos] = slerp
    # AFTER
    out = out.index_put((mask_pos,), slerp)

    if stats is not None:
        stats["slerp_angle_mean"] = omega.squeeze(-1).mean().detach()

    return out.to(embedding_matrix.dtype)


class TransparencyHead(nn.Module):
    def __init__(self, mask_token_id, trans_args):
        super().__init__()

        self.mask_token_id = mask_token_id

        # Initial scales
        init_scale = getattr(trans_args, "init_scale", 0.0)
        init_centre = getattr(trans_args, "init_centre", -0.75)
        init_steep = getattr(trans_args, "init_steep", 10 / 1.5)
        init_temperature = getattr(trans_args, "init_temperature", 1.0)

        # Scales with transformations to stay in boundaries
        self.raw_scale = nn.Parameter(
            torch.logit(torch.tensor(init_scale), eps=1e-6)
        )  # keeps in (0,1)
        self.raw_centre_neg = softplus_inv_param(-init_centre)
        self.raw_steep = softplus_inv_param(init_steep)
        self.raw_temperature = softplus_inv_param(init_temperature)

        self.mixinputs_k = getattr(trans_args, "mixinputs_k", 3)
        self.transparency_alg = getattr(
            trans_args, "transparency_alg", "mixinputs_with_topk"
        )
        self.slerp_n_iter = getattr(trans_args, "slerp_n_iter", 3)
        self.learnable = getattr(trans_args, "learnable", True)
        self.fixed_lambda = getattr(trans_args, "fixed_lambda", None)
        if self.fixed_lambda is not None:
            self.fixed_lambda = float(self.fixed_lambda)
            if not 0.0 <= self.fixed_lambda <= 1.0:
                raise ValueError(
                    f"fixed_lambda must lie in [0, 1], got {self.fixed_lambda}"
                )

        self.epsilon = 1e-6

        # Realized interpolation stats from the most recent forward (for logging
        # only; plain attrs so they never enter state_dict / EMA).
        self.last_lambda_mean = (
            torch.tensor(float(self.fixed_lambda))
            if self.fixed_lambda is not None
            else torch.tensor(0.0)
        )
        self.last_lambda_std = torch.tensor(0.0)
        self.last_slerp_angle_mean = torch.tensor(0.0)

    @property
    def scale(self):
        return torch.sigmoid(self.raw_scale)

    @property
    def centre(self):
        return -F.softplus(self.raw_centre_neg) - self.epsilon

    @property
    def steepness(self):
        return F.softplus(self.raw_steep) + self.epsilon

    @property
    def temperature(self):
        return F.softplus(self.raw_temperature) + self.epsilon

    def get_neg_entropy_and_probabilities(self, logits, temperature=1.0):
        """Get negative entropy and probabilities from logits"""
        epsilon = 1e-10
        p = torch.softmax(logits / temperature, dim=-1)  # (B,T,V)
        logp = torch.log(p + epsilon)
        neg_entropy = torch.sum(p * logp, dim=-1).to(dtype=logits.dtype)
        return neg_entropy, p

    def calculate_lambda_tensor(
        self, neg_entropy, mask_positions, current_nll=None, initial_nll=None
    ):
        """Calculate lambda tensor from negative entropy"""
        if self.fixed_lambda is not None:
            lambda_tensor = torch.full_like(
                mask_positions, fill_value=self.fixed_lambda, dtype=torch.float32
            ).to(dtype=self.raw_scale.dtype)
            lambda_tensor = torch.where(
                mask_positions, lambda_tensor, torch.zeros_like(lambda_tensor)
            )
            return lambda_tensor

        # changed

        if neg_entropy is None:
            raise ValueError(
                "neg_entropy must not be None when fixed_lambda is not set"
            )

        lambda_tensor = neg_entropy

        centre = self.centre
        if current_nll is not None and initial_nll is not None and initial_nll > 0:
            centre = self.centre * (current_nll / initial_nll)

        self.last_effective_centre = centre.detach()

        lambda_tensor = self.scale * torch.sigmoid(
            self.steepness * (lambda_tensor - centre)
        )

        # apply only on mask positions
        lambda_tensor = torch.where(
            mask_positions, lambda_tensor, torch.zeros_like(lambda_tensor)
        )
        return lambda_tensor

    def forward(
        self,
        input_ids,
        logits_prelim,
        embedding_matrix=None,
        current_nll=None,
        initial_nll=None,
    ):

        # --- 1. Get Entropy and Lambda ---
        temperature = (
            self.temperature if self.transparency_alg == "mixinputs_with_temp" else 1.0
        )
        mask_positions = input_ids == self.mask_token_id  # (B, T)

        # slerp_sm and topk never use p_full. Restrict the softmax to masked
        # positions only (avoids full (B,T,V) softmax for unmasked tokens).
        if self.transparency_alg in ("slerp_sm", "mixinputs_with_topk"):
            # GATHER: Select only the logits for masked positions
            # -----------changed------------

            masked_logits = logits_prelim[mask_positions]  # (M, V)
            neg_entropy = logits_prelim.new_zeros(input_ids.shape)
            if self.fixed_lambda is None and masked_logits.shape[0] > 0:
                neg_entropy_m, _ = self.get_neg_entropy_and_probabilities(
                    masked_logits, temperature=temperature
                )  # (M,)
                neg_entropy[mask_positions] = neg_entropy_m
            p_full = None
        else:
            if self.fixed_lambda is None:
                neg_entropy, p_full = self.get_neg_entropy_and_probabilities(
                    logits_prelim, temperature=temperature
                )
            else:
                neg_entropy = logits_prelim.new_zeros(input_ids.shape)
                p_full = None

        lambda_tensor = self.calculate_lambda_tensor(
            neg_entropy, mask_positions, current_nll, initial_nll
        )
        # AFTER- changed
        lambda_tensor = lambda_tensor.unsqueeze(-1)  # (B, T, 1)

        # Stash the realized mean / std of lambda over masked positions (live logging).
        if mask_positions.any():
            masked_lambdas = lambda_tensor.squeeze(-1)[mask_positions]
            self.last_lambda_mean = masked_lambdas.mean().detach()
            # unbiased=False so a single masked position still yields std=0 instead of NaN
            self.last_lambda_std = masked_lambdas.std(unbiased=False).detach()

        if self.transparency_alg == "slerp_sm":
            # Spherical feedback in embedding space; returns inputs_embeds (B,T,D).
            assert (
                embedding_matrix is not None
            ), "transparency_alg='slerp_sm' requires the token embedding matrix"
            stats = {}
            out = slerp_sm_feedback(
                input_ids,
                logits_prelim,
                embedding_matrix,
                self.mask_token_id,
                lambda_tensor,
                self.mixinputs_k,
                self.slerp_n_iter,
                self.epsilon,
                stats=stats,
            )
            self.last_slerp_angle_mean = stats.get("slerp_angle_mean")
            return out

        if self.transparency_alg == "mixinputs_with_topk":
            # GATHER: Select only the logits for masked positions
            # masked_logits = logits_prelim[mask_positions]

            if masked_logits.shape[0] > 0:
                # COMPUTE: Get top-k indices and probs for masked items
                topk_indices_masked, topk_probs_masked = self.get_only_topk_probs(
                    masked_logits, self.mixinputs_k
                )

                # SCATTER: Create full (B, T, k) tensors
                topk_indices = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], self.mixinputs_k),
                    dtype=topk_indices_masked.dtype,
                    device=input_ids.device,
                )
                topk_probs = torch.zeros_like(
                    topk_indices, dtype=topk_probs_masked.dtype
                )

                topk_indices[mask_positions] = topk_indices_masked
                topk_probs[mask_positions] = topk_probs_masked

            else:
                # No masks, just create empty tensors
                topk_indices = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], self.mixinputs_k),
                    dtype=torch.long,
                    device=input_ids.device,
                )
                topk_probs = torch.zeros_like(topk_indices, dtype=logits_prelim.dtype)

            # Component 1: The original one-hot token
            indices_onehot = input_ids.unsqueeze(-1)
            probs_onehot = 1.0 - lambda_tensor

            # Component 2: The top-k predictions
            indices_topk = topk_indices
            probs_topk = lambda_tensor * topk_probs  # Scale by lambda

            # Concatenate into final (B, T, k+1) tensors
            final_indices = torch.cat([indices_onehot, indices_topk], dim=-1)
            final_probs = torch.cat([probs_onehot, probs_topk], dim=-1)

            # Return the two sparse tensors as a tuple
            return (final_indices, final_probs)

        else:
            xt_one_hot = F.one_hot(input_ids, num_classes=logits_prelim.shape[-1]).to(
                logits_prelim.dtype
            )

            # p_out shape is (B, T, V)
            p_out = (1 - lambda_tensor) * xt_one_hot + lambda_tensor * p_full

            return p_out

    def get_only_topk_probs(self, logits, mixinputs_k=3):
        """
        Returns (topk_indices, topk_probs) for the M already-masked positions.
        Shape: (M, k), (M, k)  — expansion to (B, T, k) is done by the caller.
        """
        topk_logits, topk_indices = torch.topk(logits, k=mixinputs_k, dim=-1)
        topk_logits = topk_logits.to(torch.float32)

        topk_probs = torch.softmax(topk_logits, dim=-1)

        # Return the components, not the full tensor
        return topk_indices, topk_probs.to(logits.dtype)
