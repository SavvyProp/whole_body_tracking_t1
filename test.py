import torch

def f_mag_q(w: torch.Tensor) -> torch.Tensor:
    # Accept (N, E) or (E,)
    if w.ndim == 1:
        w = w.unsqueeze(0)  # (1, E)

    # Same scaling as your original
    logits    = -torch.clip(w, min=-6.0, max=6.0)  # (N, E)
    scale_lin = torch.exp(logits)                  # (N, E)
    scale_ang = scale_lin * 40.0                   # (N, E)

    # Build per-effector 6-tuple = [lin, lin, lin, ang, ang, ang]
    # Shape: (N, E, 6) so each effector's 6 entries stay contiguous
    lin3 = scale_lin.unsqueeze(-1).expand(-1, -1, 3)  # (N,E,3)
    ang3 = scale_ang.unsqueeze(-1).expand(-1, -1, 3)  # (N,E,3)
    per_eff = torch.cat([lin3, ang3], dim=-1)         # (N,E,6)

    # Flatten effector+axis to (N, E*6), then put on the diagonal
    diag_vec = per_eff.reshape(per_eff.shape[0], -1)  # (N, E*6)
    qp_q = torch.diag_embed(diag_vec)                 # (N, E*6, E*6)
    return qp_q

v1 = torch.tensor([0.1, 0.2])
out = f_mag_q(v1)
print(out)
