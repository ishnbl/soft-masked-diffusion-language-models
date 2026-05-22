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
    mu = F.normalize(vhat[..., 0, :], dim=-1, eps=eps)        # (..., D)
    w = weights.unsqueeze(-1)                                  # (..., k, 1)
    for _ in range(n_iter):
        # cos of the angle between mu and each v_i
        cos = (mu.unsqueeze(-2) * vhat).sum(-1).clamp(-1 + eps, 1 - eps)  # (..., k)
        omega = torch.acos(cos)                                # (..., k)
        sin_omega = torch.sin(omega)
        # log map at mu: ratio = omega / sin(omega) -> 1 as omega -> 0
        ratio = torch.where(sin_omega > eps,
                            omega / sin_omega.clamp_min(eps),
                            torch.ones_like(omega))
        tang = ratio.unsqueeze(-1) * (vhat - cos.unsqueeze(-1) * mu.unsqueeze(-2))  # (..., k, D)
        tau = (w * tang).sum(-2)                               # (..., D)
        tau_norm = tau.norm(dim=-1, keepdim=True)              # (..., 1)
        direction = tau / tau_norm.clamp_min(eps)
        # exp map back onto the sphere; as tau_norm -> 0 this leaves mu unchanged
        mu = torch.cos(tau_norm) * mu + torch.sin(tau_norm) * direction
        mu = F.normalize(mu, dim=-1, eps=eps)
    return mu


def slerp_sm_feedback(input_ids, logits, embedding_matrix, mask_token_id,
                      lambda_tensor, top_k, n_iter=3, eps=1e-6):
    """
    Soft-mask feedback in embedding space.

    For each masked position we SLERP between the (normalised) mask-token
    embedding and the Frechet mean of the top-k predicted token embeddings on
    the unit hypersphere. Unmasked positions keep their own token embedding.

    input_ids:        (B, T)     current (partially masked) token ids
    logits:           (B, T, V)  feedback logits from the previous pass
    embedding_matrix: (V, D)     token embedding table E
    lambda_tensor:    (B, T, 1)  SLERP weight, 0 on non-mask positions
    Returns:          (B, T, D)  soft input embeddings
    """
    compute_dtype = torch.float32
    E = embedding_matrix.to(compute_dtype)                     # (V, D)

    # Top-k tokens and renormalised weights pi (== softmax over the top-k logits).
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)   # (B, T, k)
    pi = torch.softmax(topk_logits.to(compute_dtype), dim=-1)         # (B, T, k)

    # Unit embeddings of the top-k tokens and Frechet mean mu*.
    vhat = F.normalize(E[topk_indices], dim=-1, eps=eps)       # (B, T, k, D)
    mu = frechet_mean_sphere(vhat, pi, n_iter, eps)            # (B, T, D)

    # Normalised mask embedding m_hat.
    mhat = F.normalize(E[mask_token_id], dim=-1, eps=eps)      # (D,)
    mhat = mhat.expand_as(mu)                                  # (B, T, D)

    # SLERP(m_hat, mu*, lambda).
    lam = lambda_tensor.to(compute_dtype)                      # (B, T, 1)
    cos = (mhat * mu).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)  # (B, T, 1)
    omega = torch.acos(cos)                                    # (B, T, 1)
    sin_omega = torch.sin(omega).clamp_min(eps)
    coeff_m = torch.sin((1 - lam) * omega) / sin_omega
    coeff_mu = torch.sin(lam * omega) / sin_omega
    slerp = coeff_m * mhat + coeff_mu * mu                     # (B, T, D)
    # When mask and mean are ~identical the SLERP is undefined -> return m_hat.
    slerp = torch.where(omega < eps, mhat, slerp)

    # Unmasked positions are returned unchanged (their own token embedding).
    mask_positions = (input_ids == mask_token_id).unsqueeze(-1)       # (B, T, 1)
    out = torch.where(mask_positions, slerp, E[input_ids])

    return out.to(embedding_matrix.dtype)

class TransparencyHead(nn.Module):
    """
    A transparency head that performs softmasking for text generation models.
    We only use this during training to learn the parameters, not during inference.
    During inference, we simply add the parameters to the diffusion_generate function and SM is handled by the model during generation.
    """
    def __init__(self, mask_token_id, trans_args):
        super().__init__()
            
        self.mask_token_id = mask_token_id

        init_scale = getattr(trans_args, "init_scale", 0.0)
        init_centre = getattr(trans_args, "init_centre", -0.75)
        init_steep = getattr(trans_args, "init_steep", 10/1.5)
        init_temperature = getattr(trans_args, "init_temperature", 1.0)

        self.raw_scale = nn.Parameter(torch.logit(torch.tensor(init_scale), eps=1e-6)) # keeps in (0,1)
        self.raw_centre_neg = softplus_inv_param(-init_centre)
        self.raw_steep = softplus_inv_param(init_steep)
        self.raw_temperature = softplus_inv_param(init_temperature)

        self.mixinputs_k = getattr(trans_args, "mixinputs_k", 3)
        self.transparency_alg = getattr(trans_args, "transparency_alg", "mixinputs_with_topk")
        self.slerp_n_iter = getattr(trans_args, "slerp_n_iter", 3)

        self.epsilon = 1e-6

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

    def forward(self, input_ids, logits_prelim, embedding_matrix=None):

        # First get the negative entropy to calculate lambda
        temperature = self.temperature if self.transparency_alg == "mixinputs_with_temp" else 1.0
        neg_entropy, p = self.get_neg_entropy_and_probabilities(logits_prelim, temperature=temperature)

        # Get mask positions
        mask_positions = (input_ids == self.mask_token_id)

        # calculate lambda tensor
        lambda_tensor = self.calculate_lambda_tensor(neg_entropy, mask_positions) # (B,T,1)

        if self.transparency_alg == "slerp_sm":
            # Spherical feedback in embedding space; returns inputs_embeds (B,T,D).
            assert embedding_matrix is not None, \
                "transparency_alg='slerp_sm' requires the token embedding matrix"
            return slerp_sm_feedback(
                input_ids, logits_prelim, embedding_matrix, self.mask_token_id,
                lambda_tensor, self.mixinputs_k, self.slerp_n_iter, self.epsilon)

        # Create a one-hot distribution for the original input `xt`.
        xt_one_hot = F.one_hot(input_ids, num_classes=logits_prelim.shape[-1]).to(logits_prelim.dtype)

        if self.transparency_alg == "mixinputs_with_topk":
            # Mix only top-k embeddings based on their softmax probabilities (after topk selection)
            p = self.get_only_topk_probs(logits_prelim, self.mixinputs_k)

        # Create convex combination
        p_out = (1 - lambda_tensor) * xt_one_hot \
                    + lambda_tensor * p

        return p_out

    def get_neg_entropy_and_probabilities(self, logits, temperature=1.0):
        """Get negative entropy and probabilities from logits"""
        epsilon = 1e-10
        p = torch.softmax(logits / temperature, dim=-1)   # (B,T,V)
        logp = torch.log(p + epsilon)
        neg_entropy = torch.sum(p * logp, dim=-1)
        return neg_entropy, p

    def calculate_lambda_tensor(self, neg_entropy, mask_positions):
        """Calculate lambda tensor from negative entropy"""
        if neg_entropy is None or self.scale is None:
            return None
        lambda_tensor = neg_entropy
        lambda_tensor = self.scale * torch.sigmoid(self.steepness * (lambda_tensor - self.centre))
        lambda_tensor = torch.where(mask_positions, lambda_tensor,
                                        torch.zeros_like(lambda_tensor))     
        return lambda_tensor.unsqueeze(-1)  # (B,T,1)



    def get_only_topk_probs(self, logits, mixinputs_k=3):
        """Mix only top-k embeddings based on their softmax probabilities (after topk selection)"""

        topk_logits, topk_indices = torch.topk(logits, k=mixinputs_k, dim=-1)  # (batch_size, seq_len, k)

        topk_probs = torch.softmax(topk_logits, dim=-1)  # (batch_size, seq_len, k)
        topk_sum = topk_probs.sum(dim=-1)  # (batch_size, seq_len)
        assert torch.allclose(topk_sum, torch.ones_like(topk_sum), atol=1e-1), \
            f"Top-k softmax probabilities do not sum to 1: max deviation = {(topk_sum - 1).abs().max().item()}"
    
        probs_full = torch.zeros_like(logits)                                 # (B, L, V)
        probs_full.scatter_(-1, topk_indices, topk_probs)                     # fill top-k
        assert torch.sum(probs_full > 0).item() == mixinputs_k * logits.shape[0] * logits.shape[1], \
            f"Number of non-zero entries in probs_full is incorrect: got {torch.sum(probs_full > 0).item()}, expected {mixinputs_k * logits.shape[0] * logits.shape[1]}"
        
        return probs_full
    

### ALL functions below are utility functions to attach/detach transparency head to/from a model ###

def attach_transparency(model, trans_args={}):
    mask_token_id = model.config.mask_token_id
    model.transparency = TransparencyHead(mask_token_id, trans_args)

def _base(model):
    """Get the base *HF* model that owns .transparency"""
    m = getattr(model, "get_base_model", lambda: model)()
    return m 

def transparency_head(model):
    """Get the transparency head from the model, if it exists"""
    return getattr(_base(model), "transparency", None)

def require_grad_for_th(model):
    """Require grad for transparency head parameters, except embed"""
    th = transparency_head(model)
    if th is not None:
        for p in th.parameters(recurse=False):
            p.requires_grad_(True)

def get_th_kwargs(model, verbose=False):
    """Get transparency head parameters as a dictionary to save in config"""
     # unwrap PEFT base (if present)
    th = transparency_head(model)
    if th is None:
        return {
            "transparency_alg": "none"
        }
    
    dtype  = torch.bfloat16
    th_params = {
        "transparency_alg": th.transparency_alg,
        "transparency_scale": th.scale.to(dtype).item(),
        "transparency_steepness": th.steepness.to(dtype).item(),
        "transparency_centre": th.centre.to(dtype).item(),
        "mixture_temp": th.temperature.to(dtype).item(),
        "mixinputs_k": th.mixinputs_k,
        "slerp_n_iter": th.slerp_n_iter,
        "transparency_scheduling": "none",  # we dont train with time_dependence
    }
    # if verbose:
    #     print(f"Transparency head params: {th_params}", flush=True)
    return th_params