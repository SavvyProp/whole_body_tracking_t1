import torch
from isaaclab.utils.math import quat_apply, matrix_from_quat
from torch._dynamo import disable

bodies = ['Trunk', 'H1', 'AL1', 'AR1', 'Waist', 'H2', 'AL2', 'AR2', 'Hip_Pitch_Left', 'Hip_Pitch_Right', 'AL3', 'AR3', 'Hip_Roll_Left', 'Hip_Roll_Right', 'AL4', 'AR4', 'Hip_Yaw_Left', 'Hip_Yaw_Right', 'AL5', 'AR5', 'Shank_Left', 'Shank_Right', 'AL6', 'AR6', 'Ankle_Cross_Left', 'Ankle_Cross_Right', 'left_hand_link', 'right_hand_link', 'left_foot_link', 'right_foot_link']
joints = ['AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_pitch', 'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Elbow_Yaw', 'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'Left_Wrist_Pitch', 'Right_Wrist_Pitch', 'Left_Knee_Pitch', 'Right_Knee_Pitch', 'Left_Wrist_Yaw', 'Right_Wrist_Yaw', 'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 'Left_Hand_Roll', 'Right_Hand_Roll', 'Left_Ankle_Roll', 'Right_Ankle_Roll']



TORQUE_LIMITS = torch.tensor([
    7, 
    18, 18, 
    30, 
    7, 
    18, 18, 
    45, 45, 
    18, 18, 
    25, 25, 
    18, 18, 
    25, 25, 
    18, 18, 
    60, 60, 
    18, 18, 
    24, 24, 
    18, 18, 
    15, 15
], dtype=torch.float32)

CTRL_NUM = 29
MASS = 34.634069
#SPHERE_RAD = 0.30
#SPHERE_MOI = 0.4 * MASS * SPHERE_RAD * SPHERE_RAD
#ANGULAR_INERTIA = torch.tensor(
#    [[SPHERE_MOI, 0.0, 0.0],
#     [0.0, SPHERE_MOI, 0.0],
#     [0.0, 0.0, SPHERE_MOI]],
#    dtype=torch.float32,
#)
ANGULAR_INERTIA = torch.tensor(
    [[ 2.76900149e+00,  4.50170509e-04,  3.66299529e-02],
    [ 4.50170509e-04,  2.30203655e+00, -4.42839862e-04],
    [ 3.66299529e-02, -4.42839862e-04,  5.62235551e-01]])

EEF_BODIES = ["left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"]
EEF_NUM = len(EEF_BODIES)

EEF_IDS = [bodies.index(name) for name in EEF_BODIES]

@torch.compile
def ctrl2logits(act):
    des_pos = act[:, 0:CTRL_NUM]
    des_com_vel = act[:, CTRL_NUM:CTRL_NUM + 3]
    w = act[:, CTRL_NUM + 3 : CTRL_NUM + EEF_NUM + 4]
    torque = act[:, CTRL_NUM + EEF_NUM + 4:
              CTRL_NUM * 2 + EEF_NUM + 4]
    des_com_angvel = act[:, CTRL_NUM * 2 + EEF_NUM + 4:
                CTRL_NUM * 2 + EEF_NUM + 7]
    logits = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "w": w,
        "torque": torque,
    }
    return logits

@torch.compile
def ctrl2components(act):
    logits = ctrl2logits(act)
    des_pos = logits["des_pos"]
    des_angvel = logits["des_com_angvel"] * 0.50
    des_acc = logits["des_com_vel"] * 1.0

    w = logits["w"]

    torque_logit = torch.tanh(logits["torque"] * 0.5)

    # Move torque limits onto the same device/dtype as the policy output.
    torque_limits = TORQUE_LIMITS.to(device=torque_logit.device, dtype=torque_logit.dtype)

    tau_naive = torque_limits[None, :] * torque_logit
    tau = tau_naive

    # Create torque weights on the same device/dtype as runtime tensors.
    torque_weight = torch.square(1.0 / torque_limits)

    d_gain_lin = 10.0
    d_gain_angvel = 10.0

    return {
        "des_pos": des_pos,
        "des_com_acc": des_acc,
        "des_com_angvel": des_angvel,
        "w": w,
        "torque": tau,
        "d_gain_lin": d_gain_lin,
        "d_gain_angvel": d_gain_angvel,
        "torque_weight": torque_weight
    }

def ctrl2components_ftf(act):
    des_pos = act[:, 0:CTRL_NUM]
    des_com_vel = act[:, CTRL_NUM:CTRL_NUM + 3] * 0.25
    des_com_angvel = act[:, CTRL_NUM + 3:CTRL_NUM + 6] * 0.50

    d_gain_lin = 10.0
    d_gain_angvel = 10.0

    return {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "d_gain_lin": d_gain_lin,
        "d_gain_angvel": d_gain_angvel,
    }

def ctrl2components_ftft(act):
    des_pos = act[:, 0:CTRL_NUM]
    des_com_vel = act[:, CTRL_NUM:CTRL_NUM + 3] * 0.25
    des_com_angvel = act[:, CTRL_NUM + 3:CTRL_NUM + 6] * 0.50
    des_tau = act[:, CTRL_NUM + 6:CTRL_NUM * 2 + 6]
    torque_limits = TORQUE_LIMITS.to(device=des_tau.device, dtype=des_tau.dtype)
    tau = torque_limits[None, :] * torch.tanh(des_tau * 0.5)
    torque_weight = torch.square(1.0 / torque_limits)

    d_gain_lin = 10.0
    d_gain_angvel = 10.0

    return {
        "des_pos": des_pos,
        "des_tau": tau,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "d_gain_lin": d_gain_lin,
        "d_gain_angvel": d_gain_angvel,
        "torque_weight": torque_weight
    }

@torch.compile
def make_centroidal_ag(eefpos, com_pos, base_quat, mass, i_b, grav_vec):
    """
    Vectorized version of make_centroidal_ag without Python loops.

    Args:
      eefpos: (N, E, 3)
      com_pos:(N, 3)

    Returns:
      a: (N, 6, 6*E)
      g: (6,)
    """
    r = eefpos - com_pos[:, None, :]  # (N, E, 3)
    N, E, _ = r.shape
    device, dtype = r.device, r.dtype

    # Skew-symmetric matrices S(r) for all effectors: (N, E, 3, 3)
    rx, ry, rz = r.unbind(-1)
    S = torch.zeros(N, E, 3, 3, device=device, dtype=dtype)
    S[..., 0, 1] = -rz
    S[..., 0, 2] =  ry
    S[..., 1, 0] =  rz
    S[..., 1, 2] = -rx
    S[..., 2, 0] = -ry
    S[..., 2, 1] =  rx

    # invI: compute on the active device/dtype (proper matrix inverse; inertia is diagonal).
    rot_mat = matrix_from_quat(base_quat)  # (N, 3, 3)
    #i_b = ANGULAR_INERTIA.to(device=device, dtype=dtype)
    i_w = rot_mat @ i_b @ rot_mat.transpose(-1, -2)
    invI = torch.linalg.inv(i_w)
    #invI_single = torch.linalg.inv(i_w)
    #invI = invI_single.expand(N, -1, -1).unsqueeze(1)

    # Bottom block per effector: [invI @ S, invI] -> (N, E, 3, 6)
    bot_left = invI.view(N, 1, 3, 3).expand(N, E, 3, 3) @ S                         # (N, E, 3, 3)
    bot_right = invI.view(N, 1, 3, 3).expand(N, E, 3, 3)         # (N, E, 3, 3)
    f_bot = torch.cat([bot_left, bot_right], dim=-1)  # (N, E, 3, 6)

    # Top block per effector: [I/M, 0] -> (N, E, 3, 6)
    I3 = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).expand(N, 3, 3)
    f_top_base = torch.cat([I3 / mass[:, None, None], torch.zeros_like(I3)], dim=-1)  # (3, 6)
    f_top = f_top_base.view(N, 1, 3, 6).expand(N, E, 3, 6)

    # Full per-effector 6x6 block: (N, E, 6, 6)
    f_block = torch.cat([f_top, f_bot], dim=-2)  # (N, E, 6, 6)

    # Concatenate horizontally across effectors: (N, 6, 6*E)
    a = f_block.permute(0, 2, 1, 3).reshape(N, 6, E * 6)

    #g = eefpos.new_tensor([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])  # (6,)
    g_base = torch.tensor([0.0, 0.0, -9.81, 0.0, 0.0, 0.0], device=eefpos.device, dtype=eefpos.dtype)
    g = g_base[None, :] + grav_vec
    g = g * 9.81 / torch.norm(g, dim=-1, keepdim=True)
    return a, g

@torch.compile
def f_mag_q(w: torch.Tensor) -> torch.Tensor:
    # Accept (N, E) or (E,)
    if w.ndim == 1:
        w = w.unsqueeze(0)  # (1, E)

    # Same scaling as your original
    logits    = -torch.clip(w, min=-10.0, max=10.0)  # (N, E)
    scale_lin = torch.exp(logits)                  # (N, E)
    scale_ang = scale_lin * 10.0                   # (N, E)

    # Build per-effector 6-tuple = [lin, lin, lin, ang, ang, ang]
    # Shape: (N, E, 6) so each effector's 6 entries stay contiguous
    lin3 = scale_lin.unsqueeze(-1).expand(-1, -1, 3)  # (N,E,3)
    ang3 = scale_ang.unsqueeze(-1).expand(-1, -1, 3)  # (N,E,3)
    per_eff = torch.cat([lin3, ang3], dim=-1)         # (N,E,6)

    # Flatten effector+axis to (N, E*6), then put on the diagonal
    diag_vec = per_eff.reshape(per_eff.shape[0], -1)  # (N, E*6)
    qp_q = torch.diag_embed(diag_vec)                 # (N, E*6, E*6)
    return qp_q

@torch.compile
def joint_torque_q(jacs: torch.Tensor, tau_ref: torch.Tensor, w: torch.Tensor | None = None):
    """
    jacs:    (N, 6*EEF_NUM, 6+CTRL_NUM) or (6*EEF_NUM, 6+CTRL_NUM)
    tau_ref: (N, CTRL_NUM)              or (CTRL_NUM,)
    w:       Optional weights for ALL dofs (base+joint):
             (N, 6+CTRL_NUM) or (6+CTRL_NUM,). Only the joint portion (last CTRL_NUM)
             is used here since J_j excludes the 6 base dofs.

    Returns:
      big_q:   (N, 6*EEF_NUM, 6*EEF_NUM) = J_j @ W @ J_j^T
      small_q: (N, 6*EEF_NUM)            = J_j @ (W @ tau_ref)
    where J_j = -jacs[..., :, 6:]  (exclude the 6 base dofs)
    and W is diagonal formed from w[..., 6:].

    Notes:
      Implemented without explicitly constructing W/diag matrices:
        big_q = (J_j * wj) @ (J_j)^T
        small_q = J_j @ (tau_ref * wj)
    """
    device, dtype = jacs.device, jacs.dtype

    # Normalize jacs to (N, F, 6+CTRL)
    if jacs.dim() == 2:
        jacs = jacs.unsqueeze(0)
    jacs = jacs.to(dtype=dtype)
    N, F, _ = jacs.shape

    # J_j: (N, F, CTRL)
    J_j = -jacs[..., :, 6:]
    CTRL = J_j.shape[-1]

    # Normalize tau_ref to (N, CTRL)
    if tau_ref.dim() == 1:
        tau_ref = tau_ref.unsqueeze(0)
    else:
        tau_ref = tau_ref.reshape(-1, tau_ref.shape[-1])
    tau_ref = tau_ref.to(device=device, dtype=dtype)

    if tau_ref.shape[-1] != CTRL:
        raise ValueError(f"CTRL dim mismatch: J_j has {CTRL} but tau_ref has {tau_ref.shape[-1]}")

    # Expand or validate batch
    if tau_ref.shape[0] == 1 and N > 1:
        tau_ref = tau_ref.expand(N, -1)
    elif tau_ref.shape[0] != N:
        raise ValueError(f"Batch mismatch: jacs batch {N} vs tau_ref batch {tau_ref.shape[0]}")

    # --- weights handling (optional) ---
    # w is defined over (6+CTRL); only last CTRL apply to J_j columns.
    if w is None:
        # Unweighted case
        big_q = J_j @ J_j.transpose(-1, -2)
        small_q = torch.bmm(J_j, tau_ref.unsqueeze(-1)).squeeze(-1)
        return big_q, small_q

    # Ensure weights are on the same device/dtype as jacs.
    wj = w
    if wj.dim() == 1:
        wj = wj.unsqueeze(0)
    else:
        wj = wj.reshape(-1, wj.shape[-1])
    wj = wj.to(device=device, dtype=dtype)

    # big_q = J_j @ W @ J_j^T  == (J_j * w_j) @ (J_j)^T
    Jw = J_j * wj.unsqueeze(-2)  # (N, F, CTRL)
    big_q = Jw @ J_j.transpose(-1, -2)

    # small_q = J_j @ (W @ tau_ref)  == J_j @ (tau_ref * wj)
    tau_w = tau_ref * wj  # (N, CTRL)
    small_q = torch.bmm(J_j, tau_w.unsqueeze(-1)).squeeze(-1)  # (N, F)

    return big_q, small_q

@torch.compile
def centroidal_qacc_cons(big_a, g, com_ref):
    lhs = big_a
    rhs = com_ref - g
    return lhs, rhs

@torch.compile
def schur_solve(
    qp_q: torch.Tensor,
    qp_c: torch.Tensor,
    cons_lhs: torch.Tensor,
    cons_rhs: torch.Tensor,
    reg: float = 1e-6,
):
    """
    qp_q:     (..., F, F)
    qp_c:     (..., F)
    cons_lhs: (..., M, F)   (A)
    cons_rhs: (..., M)      (b)

    Returns:
      x: (..., F)
    """
    device, dtype = qp_q.device, qp_q.dtype
    qp_c = qp_c.to(device=device, dtype=dtype)
    cons_lhs = cons_lhs.to(device=device, dtype=dtype)
    cons_rhs = cons_rhs.to(device=device, dtype=dtype)

    squeeze_out = False
    if qp_q.dim() == 2:
        qp_q = qp_q.unsqueeze(0)
        qp_c = qp_c.unsqueeze(0)
        cons_lhs = cons_lhs.unsqueeze(0)
        cons_rhs = cons_rhs.unsqueeze(0)
        squeeze_out = True

    batch_shape = qp_q.shape[:-2]
    F = qp_q.shape[-1]
    M = cons_lhs.shape[-2]

    # Symmetrize Q
    Q = 0.5 * (qp_q + qp_q.transpose(-1, -2))

    # Optional Tikhonov regularization on Q
    if reg > 0.0:
        I = torch.eye(F, device=device, dtype=dtype).expand(*batch_shape, F, F)
        Q = Q + reg * I

    A = cons_lhs                      # (..., M, F)
    AT = A.transpose(-1, -2)          # (..., F, M)
    c = qp_c                          # (..., F)
    b = cons_rhs                      # (..., M)

    # Factor Q once: LU (works for indefinite too)
    LU, pivots, infoQ = torch.linalg.lu_factor_ex(Q, pivot=True, check_errors=False)

    def solve_Q(B: torch.Tensor) -> torch.Tensor:
        # solves Q X = B
        return torch.linalg.lu_solve(LU, pivots, B)

    # Compute Q^{-1} A^T and Q^{-1} c
    Qinv_AT = solve_Q(AT)                              # (..., F, M)
    Qinv_c = solve_Q(c.unsqueeze(-1)).squeeze(-1)      # (..., F)

    # Schur matrix S = A Q^{-1} A^T  and rhs = A Q^{-1} c - b
    S = A @ Qinv_AT                                    # (..., M, M)
    rhs_lam = (A @ Qinv_c.unsqueeze(-1)).squeeze(-1) - b   # (..., M)

    # Solve for lambda (small MxM system)
    lam, _ = torch.linalg.solve_ex(S, rhs_lam.unsqueeze(-1), check_errors=False)
    lam = lam.squeeze(-1)                              # (..., M)

    # Recover x = Q^{-1}(c - A^T lambda)
    x = solve_Q((c - (AT @ lam.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)

    if squeeze_out:
        x = x.squeeze(0)
    return x


def ft_ref(
    eefpos_, com_pos, jacs_, tau_ref, com_ref, w, torque_weight, base_quat, mass, i_b, grav_vec, nle
):
    # Concat the unaccounted force component
    ctrl_num = tau_ref.shape[-1]
    unaccounted_jac = torch.zeros(
        (jacs_.shape[0], 6, ctrl_num + 6), device = jacs_.device
    )
    unaccounted_jac[:, :6, :6] = torch.eye(6, device = jacs_.device)
    jacs = torch.cat([unaccounted_jac, jacs_], dim = 1)
    eefpos = torch.cat([
        com_pos[:, None, :], eefpos_
    ], dim = 1)

    # Ensure weights tensor matches eefpos device/dtype.
    weights = torch.tensor([1e-3, 1e1], device=eefpos.device, dtype=eefpos.dtype)
    a, g = make_centroidal_ag(eefpos, com_pos, base_quat, mass, i_b, grav_vec)

    qp_q_ = f_mag_q(w)  # (N, 6*EEF_NUM, 6*EEF_NUM)
    qp_q_ = qp_q_ * weights[0]
    
    jt_q_big, jt_q_small = joint_torque_q(jacs, tau_ref, torque_weight)
    jt_q_big = jt_q_big * weights[1]

    qp_q = qp_q_ + jt_q_big
    qp_c = jt_q_small * weights[1]

    cons_lhs, cons_rhs = centroidal_qacc_cons(a, g, com_ref)

    f = schur_solve(qp_q, qp_c, cons_lhs, cons_rhs)

    candidate_tau = -jacs[..., :, 6:].transpose(-1, -2) @ f[..., None]
    candidate_tau = candidate_tau.squeeze(-1)
    candidate_tau = candidate_tau + nle

    # Clamp using torque limits on the same device/dtype as candidate_tau.
    torque_limits = TORQUE_LIMITS.to(device=candidate_tau.device, dtype=candidate_tau.dtype)
    tau = torch.clamp(candidate_tau, min=-torque_limits[None, :], max=torque_limits[None, :])

    f = f[:, 6:] # remove unaccounted force
    info = {
        "f": f,
        "candidate_tau": candidate_tau,
        "w": w,
    }
    return tau, info

def highlvlPD(base_quat, base_angvel, 
              angvel_gain,
              com_acc, des_angvel,
              ):
    q_wb = base_quat
    global_des_acc = quat_apply(q_wb, com_acc)
    global_des_angvel = quat_apply(q_wb, des_angvel)

    #com_acc = com_acc#lin_gain * (global_des_vel - com_vel)

    # com_acc should be clipped to a max of 5

    acc_mag = torch.linalg.norm(global_des_acc, dim=-1, keepdim=True)
    max_acc = 5.0
    new_acc_mag = torch.clamp(acc_mag, max=max_acc)
    global_des_acc = global_des_acc * (new_acc_mag / (acc_mag + 1e-6))
    #com_acc = torch.clamp(com_acc, min=-3.0, max=3.0)

    com_angvel = base_angvel
    ang_acc = angvel_gain * (global_des_angvel - com_angvel)

    return global_des_acc, ang_acc, global_des_acc, global_des_angvel

def step(com_pos,
         jacs,
         eefpos,
         base_quat, base_angvel,
         action, nle, lcc_rand):
    comp_dict = ctrl2components(action)
    #com_vel_ = com_vel + lcc_rand["com_vel"]
    base_angvel_ = base_angvel + lcc_rand["com_angvel"]
    com_acc, ang_acc, global_vel, global_angvel = highlvlPD(
        base_quat, base_angvel_, comp_dict["d_gain_angvel"],
        comp_dict["des_com_acc"], comp_dict["des_com_angvel"],
    )

    idx = torch.as_tensor(EEF_IDS, device=jacs.device, dtype=torch.long)
    selected_jacs = jacs.index_select(1, idx)                 # (N, EEF_NUM, 6, D)
    jacs_ = selected_jacs.reshape(selected_jacs.size(0), -1, selected_jacs.size(-1))  # (N, 6*EEF_NUM, D)
    eefpos_ = eefpos.index_select(1, idx)                 # (N, EEF_NUM, 3)

    # Add offsets to pos
    eefpos_offset = lcc_rand["pos"][:, 1:, :]
    eefpos_0 = eefpos_ + eefpos_offset
    com_pos_ = com_pos + lcc_rand["pos"][:, 0, :]

    # Modify jacs
    jacs_0 = jacs_ * lcc_rand["jac_fac"]

    # Modify mass
    mass = MASS * lcc_rand["mass_fac"]
    i_b = ANGULAR_INERTIA.to(device = com_pos.device).view(1, 3, 3) * lcc_rand["i_fac"] 

    tau, info = ft_ref(
        eefpos_0, com_pos_, jacs_0,
        comp_dict["torque"],
        torch.cat([com_acc, ang_acc], dim=-1),
        comp_dict["w"],
        comp_dict["torque_weight"],
        base_quat, mass, i_b, lcc_rand["grav_vec"], nle
    )
    info["com_vel"] = global_vel
    info["com_angvel"] = global_angvel
    info["com_acc"] = com_acc
    info["com_angacc"] = ang_acc
    return comp_dict["des_pos"], tau, info

def ftf_step(com_pos, 
             jacs, eefpos,
             base_quat, base_angvel,
             action, contact_state, nle, lcc_rand):
    # Variant of ft step with modified action
    # fixed contact state with weight of 2^10 for contacting and 2^-10 for non contacting
    # no reference torque
    comp_dict = ctrl2components_ftf(action)
    #com_vel_ = com_vel + lcc_rand["com_vel"]
    base_angvel_ = base_angvel + lcc_rand["com_angvel"]
    com_acc, ang_acc, global_vel, global_angvel = highlvlPD(
        base_quat, base_angvel_, comp_dict["d_gain_angvel"],
        comp_dict["des_com_acc"], comp_dict["des_com_angvel"],
    )

    idx = torch.as_tensor(EEF_IDS, device=jacs.device, dtype=torch.long)
    selected_jacs = jacs.index_select(1, idx)                 # (N, EEF_NUM, 6, D)
    jacs_ = selected_jacs.reshape(selected_jacs.size(0), -1, selected_jacs.size(-1))  # (N, 6*EEF_NUM, D)
    eefpos_ = eefpos.index_select(1, idx)                 # (N, EEF_NUM, 3)

    # Add offsets to pos
    eefpos_offset = lcc_rand["pos"][:, 1:, :]
    eefpos_0 = eefpos_ + eefpos_offset
    com_pos_ = com_pos + lcc_rand["pos"][:, 0, :]

    # Modify jacs
    jacs_0 = jacs_ * lcc_rand["jac_fac"]

    # Modify mass
    mass = MASS * lcc_rand["mass_fac"]
    i_b = ANGULAR_INERTIA.to(device = com_pos.device).view(1, 3, 3) * lcc_rand["i_fac"] 

    a, g = make_centroidal_ag(eefpos_0, com_pos_, base_quat, mass, i_b, lcc_rand["grav_vec"])
    w = contact_state * 20.0 - 10.0
    qp_q = f_mag_q(w)
    qp_c = torch.zeros((com_pos.shape[0], qp_q.shape[-1]), device=com_pos.device, dtype=com_pos.dtype)

    f = schur_solve(qp_q, qp_c, a, g)
    candidate_tau = -jacs_0[..., :, 6:].transpose(-1, -2) @ f[..., None]
    candidate_tau = candidate_tau.squeeze(-1)
    candidate_tau = candidate_tau + nle

    torque_limits = TORQUE_LIMITS.to(device=candidate_tau.device, dtype=candidate_tau.dtype)
    tau = torch.clamp(candidate_tau, min=-torque_limits[None, :], max=torque_limits[None, :])

    info = {
        "f": f,
        "candidate_tau": candidate_tau,
        "w": w,
        "com_vel": global_vel,
        "com_angvel": global_angvel,
        "com_acc": com_acc,
        "com_angacc": ang_acc,
    }
    return comp_dict["des_pos"], tau, info


def ftft_step(com_pos, 
             jacs, eefpos,
             base_quat, base_angvel,
             action, contact_state, nle, lcc_rand):
    # Variant of ft step with modified action
    # fixed contact state with weight of 2^10 for contacting and 2^-10 for non contacting
    # no reference torque
    comp_dict = ctrl2components_ftft(action)
    #com_vel_ = com_vel + lcc_rand["com_vel"]
    base_angvel_ = base_angvel + lcc_rand["com_angvel"]
    com_acc, ang_acc, global_vel, global_angvel = highlvlPD(
        base_quat, base_angvel_, comp_dict["d_gain_angvel"],
        comp_dict["des_com_vel"], comp_dict["des_com_angvel"],
    )

    idx = torch.as_tensor(EEF_IDS, device=jacs.device, dtype=torch.long)
    selected_jacs = jacs.index_select(1, idx)                 # (N, EEF_NUM, 6, D)
    jacs_ = selected_jacs.reshape(selected_jacs.size(0), -1, selected_jacs.size(-1))  # (N, 6*EEF_NUM, D)
    eefpos_ = eefpos.index_select(1, idx)                 # (N, EEF_NUM, 3)

    # Add offsets to pos
    eefpos_offset = lcc_rand["pos"][:, 1:, :]
    eefpos_0 = eefpos_ + eefpos_offset
    com_pos_ = com_pos + lcc_rand["pos"][:, 0, :]

    # Modify jacs
    jacs_0 = jacs_ * lcc_rand["jac_fac"]

    # Modify mass
    mass = MASS * lcc_rand["mass_fac"]
    i_b = ANGULAR_INERTIA.to(device = com_pos.device).view(1, 3, 3) * lcc_rand["i_fac"] 

    weights = torch.tensor([1e-3, 1e1], device=eefpos.device, dtype=eefpos.dtype)
    a, g = make_centroidal_ag(eefpos_0, com_pos_, base_quat, mass, i_b, lcc_rand["grav_vec"])

    w = contact_state * 20.0 - 10.0
    qp_q_ = f_mag_q(w)  # (N, 6*EEF_NUM, 6*EEF_NUM)
    qp_q_ = qp_q_ * weights[0]
    
    jt_q_big, jt_q_small = joint_torque_q(jacs, comp_dict["tau"], comp_dict["torque_weight"])
    jt_q_big = jt_q_big * weights[1]

    qp_q = qp_q_ + jt_q_big
    qp_c = jt_q_small * weights[1]
    #qp_c = torch.zeros((com_pos.shape[0], qp_q.shape[-1]), device=com_pos.device, dtype=com_pos.dtype)

    f = schur_solve(qp_q, qp_c, a, g)
    candidate_tau = -jacs_0[..., :, 6:].transpose(-1, -2) @ f[..., None]
    candidate_tau = candidate_tau.squeeze(-1)
    candidate_tau = candidate_tau + nle

    torque_limits = TORQUE_LIMITS.to(device=candidate_tau.device, dtype=candidate_tau.dtype)
    tau = torch.clamp(candidate_tau, min=-torque_limits[None, :], max=torque_limits[None, :])

    info = {
        "f": f,
        "candidate_tau": candidate_tau,
        "w": w,
        "com_vel": global_vel,
        "com_angvel": global_angvel,
        "com_acc": com_acc,
        "com_angacc": ang_acc,
    }
    return comp_dict["des_pos"], tau, info

try:
    jit_step = torch.compile(step)
except Exception as _e:
    print("[INFO] torch.compile disabled; using eager mode:", _e)