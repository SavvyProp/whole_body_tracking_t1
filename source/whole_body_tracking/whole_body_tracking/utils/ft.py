import torch
from isaaclab.utils.math import quat_apply

bodies = ['Trunk', 'H1', 'AL1', 'AR1', 'Waist', 'H2', 'AL2', 'AR2', 'Hip_Pitch_Left', 'Hip_Pitch_Right', 'AL3', 'AR3', 'Hip_Roll_Left', 'Hip_Roll_Right', 'left_hand_link', 'right_hand_link', 'Hip_Yaw_Left', 'Hip_Yaw_Right', 'Shank_Left', 'Shank_Right', 'Ankle_Cross_Left', 'Ankle_Cross_Right', 'left_foot_link', 'right_foot_link']
joints = ['AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_pitch', 'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Elbow_Yaw', 'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'Left_Knee_Pitch', 'Right_Knee_Pitch', 'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 'Left_Ankle_Roll', 'Right_Ankle_Roll']

CTRL_NUM = 23
EEF_NUM = 4

TORQUE_LIMITS = torch.tensor([
    7, 18, 18, 30, 7, 18, 18, 45, 45, 18, 18, 30, 30, 18, 18, 30, 30, 60, 60, 20, 20, 15, 15
], device = "cuda")

MASS = 31.614357
ANGULAR_INERTIA = torch.tensor(
    [[ 2.77498525e+00,  5.36123413e-04,  2.12637797e-01],
 [ 5.36123413e-04,  2.64427940e+00, -2.98730940e-03],
 [ 2.12637797e-01, -2.98730940e-03,  4.91490757e-01]], device = "cuda")
INV_ANGULAR_INERTIA = torch.linalg.inv(ANGULAR_INERTIA)

EEF_BODIES = ["left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"]

EEF_IDS = [bodies.index(name) for name in EEF_BODIES]

def ctrl2logits(act):
    des_pos = act[:, 0:CTRL_NUM]
    des_com_vel = act[:, CTRL_NUM:CTRL_NUM + 3]
    des_com_angvel = act[:, CTRL_NUM + 3 : CTRL_NUM + 6]
    w = act[:, CTRL_NUM + 6 : CTRL_NUM + EEF_NUM + 6]
    torque = act[:, CTRL_NUM + EEF_NUM + 6:
              CTRL_NUM * 2 + EEF_NUM + 6]
    d_gain = act[:, -2:]
    logits = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "w": w,
        "torque": torque,
        "d_gain": d_gain
    }
    return logits

def ctrl2components(act, joint_vel):
    logits = ctrl2logits(act)
    des_pos = logits["des_pos"]

    des_angvel = logits["des_com_angvel"] * 0.20
    #des_angvel_mag = torch.norm(des_angvel, dim =-1, keepdim=True)
    #des_angvel_mag_clipped = torch.clamp(des_angvel_mag, max = 2.0)
    #des_angvel = des_angvel * (des_angvel_mag_clipped / (1e-6 + des_angvel_mag))
    
    #des_com_vel = logits["des_com_vel"] * 0.05
    #des_vel_mag = torch.norm(des_com_vel, dim =-1, keepdim=True)
    #des_vel_mag_clipped = torch.clamp(des_vel_mag, max = 3.0)
    #des_vel = des_com_vel * (des_vel_mag_clipped / (1e-6 + des_vel_mag))
    
    des_vel = logits["des_com_vel"] * 0.25

    w = logits["w"]

    torque_logit = torch.tanh(logits["torque"] * 0.5)
    torque_limits = TORQUE_LIMITS.to(torque_logit.device)
    tau_naive = torque_limits[None, :] * torque_logit
    spd_fac = torch.clip(torch.abs(joint_vel), min = 0.0, max = 10.0) / 10.0
    sign = torch.where(joint_vel * torque_logit >= 0, 1.0, 0.0)
    tau = tau_naive * (1.0 - spd_fac[None, :] * sign)

    d_gain_lin = torch.tanh(logits["d_gain"][:, 0]) * 3.0 + 4.0
    #d_gain_lin = jnp.tanh(logits["d_gain"][0]) * 6.0 + 7.0
    d_gain_angvel = torch.tanh(logits["d_gain"][:, 1]) * 0.10 + 0.11

    return {
        "des_pos": des_pos,
        "des_com_vel": des_vel,
        "des_com_angvel": des_angvel,
        "w": w,
        "torque": tau,
        "d_gain_lin": d_gain_lin,
        "d_gain_angvel": d_gain_angvel
    }

def make_centroidal_ag(
    eefpos, com_pos
):
    r = eefpos - com_pos[:, None, :]
    f_blocks = []
    batch_invI = INV_ANGULAR_INERTIA.expand(r.shape[0], -1, -1)  # (N,3,3)
    for i in range(EEF_NUM):
        v = r[:, i, :]  # (N, 3)
        S = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
        S[:, 0, 1] = -v[:, 2]; S[:, 0, 2] =  v[:, 1]
        S[:, 1, 0] =  v[:, 2]; S[:, 1, 2] = -v[:, 0]
        S[:, 2, 0] = -v[:, 1]; S[:, 2, 1] =  v[:, 0]
        f_top = torch.cat([
            torch.eye(3, device=v.device) / MASS,
            torch.zeros((3, 3), device=v.device)
        ], dim=1)  # (N, 3, 6)
        f_top = f_top.expand(v.shape[0], -1, -1)
        f_bot = torch.cat([batch_invI @ S, batch_invI], dim=2)
        f_block = torch.cat([f_top, f_bot], dim=1)  # (N, 6,6)
        f_blocks.append(f_block)
    a = torch.cat(f_blocks, dim=2)  # (N, 6, 6*EEF_NUM)
    g = torch.tensor([0, 0, -9.81, 0, 0, 0], device=eefpos.device)  # (6,)
    return a, g

def f_mag_q(w):
    # w: (N, EEF_NUM) or (EEF_NUM,)
    logits = -torch.clip(w, min=-6.0, max=6.0)          # (N, EEF_NUM)
    scale_lin = torch.exp(logits)                        # (N, EEF_NUM)
    scale_ang = scale_lin * 40.0                         # (N, EEF_NUM)

    F_size = EEF_NUM * 6
    qp_q = torch.eye(F_size, device=w.device, dtype=w.dtype).unsqueeze(0).expand(w.shape[0], -1, -1).clone()

    for i in range(EEF_NUM):
        r0 = i * 6
        r3 = r0 + 3
        r6 = r0 + 6
        s_lin = scale_lin[:, i].view(-1, 1, 1)          # (N,1,1)
        s_ang = scale_ang[:, i].view(-1, 1, 1)          # (N,1,1)
        qp_q[:, r0:r3, r0:r3] *= s_lin                  # translational 3x3
        qp_q[:, r3:r6, r3:r6] *= s_ang                  # rotational 3x3

    return qp_q

def joint_torque_q(jacs: torch.Tensor, tau_ref: torch.Tensor):
    """
    jacs:    (N, 6*EEF_NUM, 6+CTRL_NUM) or (6*EEF_NUM, 6+CTRL_NUM)
    tau_ref: (N, CTRL_NUM)              or (CTRL_NUM,)

    Returns:
      big_q:   (N, 6*EEF_NUM, 6*EEF_NUM) = J_j @ J_j^T
      small_q: (N, 6*EEF_NUM)            = J_j @ tau_ref
    where J_j = -jacs[..., :, 6:]  (exclude the 6 base dofs)
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
        tau_ref = tau_ref.unsqueeze(0)          # (1, CTRL)
    else:
        tau_ref = tau_ref.reshape(-1, tau_ref.shape[-1])  # (N?, CTRL)
    tau_ref = tau_ref.to(device=device, dtype=dtype)

    if tau_ref.shape[-1] != CTRL:
        raise ValueError(f"CTRL dim mismatch: J_j has {CTRL} but tau_ref has {tau_ref.shape[-1]}")

    # Expand or validate batch
    if tau_ref.shape[0] == 1 and N > 1:
        tau_ref = tau_ref.expand(N, -1)
    elif tau_ref.shape[0] != N:
        raise ValueError(f"Batch mismatch: jacs batch {N} vs tau_ref batch {tau_ref.shape[0]}")

    # big_q: (N, F, F)
    big_q = J_j @ J_j.transpose(-1, -2)

    # small_q: (N, F) using bmm to avoid broadcasting surprises
    small_q = torch.bmm(J_j, tau_ref.unsqueeze(-1)).squeeze(-1)

    return big_q, small_q

def centroidal_qacc_cons(big_a, g, com_ref):
    lhs = big_a
    rhs = com_ref - g
    return lhs, rhs

def schur_solve(qp_q: torch.Tensor, qp_c: torch.Tensor, cons_lhs: torch.Tensor, cons_rhs: torch.Tensor, reg: float = 0.0):
    """
    qp_q:    (..., F, F)
    qp_c:    (..., F)
    cons_lhs:(..., M, F)   (A)
    cons_rhs:(..., M)      (b)

    Returns:
      sol:   (..., F)
    """
    # Ensure common device/dtype
    device = qp_q.device
    dtype = qp_q.dtype
    qp_c = qp_c.to(device=device, dtype=dtype)
    cons_lhs = cons_lhs.to(device=device, dtype=dtype)
    cons_rhs = cons_rhs.to(device=device, dtype=dtype)

    # Add batch dim if unbatched
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

    # Optional Tikhonov regularization on Q block
    if reg > 0.0:
        I_F = torch.eye(F, device=device, dtype=dtype).expand(*batch_shape, F, F)
        Q = Q + reg * I_F

    A = cons_lhs
    c = qp_c
    b = cons_rhs

    # Build KKT = [[Q, A^T],
    #              [A, 0   ]]
    AT = A.transpose(-1, -2)                                # (..., F, M)
    Z = Q.new_zeros(*batch_shape, M, M)                     # (..., M, M)
    top = torch.cat([Q, AT], dim=-1)                        # (..., F, F+M)
    bot = torch.cat([A, Z], dim=-1)                         # (..., M, F+M)
    KKT = torch.cat([top, bot], dim=-2)                     # (..., F+M, F+M)

    rhs = torch.cat([c, b], dim=-1)                         # (..., F+M)

    sol_all = torch.linalg.solve(KKT, rhs.unsqueeze(-1)).squeeze(-1)

    sol = sol_all[..., :F]                                   # (..., F)

    if squeeze_out:
        sol = sol.squeeze(0)
    return sol

def ft_ref(
    eefpos, com_pos, jacs, tau_ref, com_ref, w, debug = False
):
    weights = torch.tensor([1e-3, 1e-2], device=eefpos.device)
    a, g = make_centroidal_ag(eefpos, com_pos)

    qp_q = f_mag_q(w)  # (N, 6*EEF_NUM, 6*EEF_NUM)
    qp_q = qp_q * weights[0]
    jt_q_big, jt_q_small = joint_torque_q(jacs, tau_ref)
    jt_q_big = jt_q_big * weights[1]

    qp_q += jt_q_big
    qp_c = jt_q_small * weights[1]

    # Make constraints

    cons_lhs, cons_rhs = centroidal_qacc_cons(a, g, com_ref)

    f = schur_solve(qp_q, qp_c, cons_lhs, cons_rhs)

    candidate_tau = -jacs[..., :, 6:].transpose(-1, -2) @ f[..., None]
    candidate_tau = candidate_tau.squeeze(-1)

    t = torch.where(candidate_tau > TORQUE_LIMITS[None, :],
                    TORQUE_LIMITS[None, :], -TORQUE_LIMITS[None, :])
    d = (t - tau_ref) / (candidate_tau - tau_ref)
    scaling_fac = torch.clamp(d, min=0.0, max=1.0)
    scaling_fac = torch.where(candidate_tau.abs() <= TORQUE_LIMITS[None, :],
                              1.0, scaling_fac)

    min_scaling_fac, _ = torch.min(scaling_fac, dim = -1, keepdim = True)
    tau = tau_ref * (1 - min_scaling_fac) + candidate_tau * min_scaling_fac
    

    if debug:
        return tau, f
    return tau

def highlvlPD(base_quat, base_angvel, 
              lin_gain, angvel_gain,
              des_vel, des_angvel,
              com_vel, w):
    default_acc = torch.tensor([0, 0, -9.81], device=base_quat.device)
    default_angacc = torch.tensor([0, 0, 0], device=base_quat.device)
    num_contacts = torch.clamp(torch.sum(torch.sigmoid(w * 2), axis = -1), min = 0.0, max = 1.0)
    q_wb = base_quat
    global_des_vel = quat_apply(q_wb, des_vel)

    com_acc = lin_gain[:, None] * (global_des_vel - com_vel)

    com_angvel = base_angvel
    ang_acc = angvel_gain[:, None] * (des_angvel - com_angvel)

    com_acc = com_acc * (1.0 - num_contacts[:, None]) + default_acc[None, :] * num_contacts[:, None]
    ang_acc = ang_acc * (1.0 - num_contacts[:, None]) + default_angacc[None, :] * num_contacts[:, None]
    return com_acc, ang_acc

def step(com_pos, com_vel,
         jacs,
         eefpos,
         base_quat, base_angvel, joint_vel,
         action):
    comp_dict = ctrl2components(action, joint_vel)
    com_acc, ang_acc = highlvlPD(
        base_quat, base_angvel,
        comp_dict["d_gain_lin"], comp_dict["d_gain_angvel"],
        comp_dict["des_com_vel"], comp_dict["des_com_angvel"],
        com_vel, comp_dict["w"]
    )

    idx = torch.as_tensor(EEF_IDS, device=jacs.device, dtype=torch.long)
    selected_jacs = jacs.index_select(1, idx)                 # (N, EEF_NUM, 6, D)
    jacs_ = selected_jacs.reshape(selected_jacs.size(0), -1, selected_jacs.size(-1))  # (N, 6*EEF_NUM, D)
    eefpos_ = eefpos.index_select(1, idx)                 # (N, EEF_NUM, 3)
    tau = ft_ref(
        eefpos_, com_pos, jacs_,
        comp_dict["torque"],
        torch.cat([com_acc, ang_acc], dim=-1),
        comp_dict["w"]
    )
    #torque_limits = TORQUE_LIMITS.to(tau.device, tau.dtype)
    #tau = torch.clamp(tau, min=-torque_limits[None, :], max=torque_limits[None, :])
    return comp_dict["des_pos"], tau

try:
    jit_step = torch.compile(step)
    # You can also compile other hot helpers if desired:
    # ft_ref = torch.compile(ft_ref, mode="max-autotune", fullgraph=False)
except Exception as _e:
    print("[INFO] torch.compile disabled; using eager mode:", _e)

def ft_rew_info(com_pos, com_vel,
         jacs,
         eefpos,
         base_quat, base_angvel, joint_vel,
         action):
    logits = ctrl2logits(action)
    comp_dict = ctrl2components(action, joint_vel)
    com_acc, ang_acc = highlvlPD(
        base_quat, base_angvel,
        comp_dict["d_gain_lin"], comp_dict["d_gain_angvel"],
        comp_dict["des_com_vel"], comp_dict["des_com_angvel"],
        com_vel, comp_dict["w"]
    )

    idx = torch.as_tensor(EEF_IDS, device=jacs.device, dtype=torch.long)
    selected_jacs = jacs.index_select(1, idx)                 # (N, EEF_NUM, 6, D)
    jacs_ = selected_jacs.reshape(selected_jacs.size(0), -1, selected_jacs.size(-1))  # (N, 6*EEF_NUM, D)
    eefpos_ = eefpos.index_select(1, idx)                 # (N, EEF_NUM, 3)
    tau, f = ft_ref(
        eefpos_, com_pos, jacs_,
        comp_dict["torque"],
        torch.cat([com_acc, ang_acc], dim=-1),
        comp_dict["w"], debug=True
    )
    debug_dict = {
        "ff_tau": tau.reshape(-1, CTRL_NUM),
        "f": f,
        "des_pos": logits["des_pos"],
    }
    return {
        "debug": debug_dict,
        "logits": ctrl2logits(action),
        "components": ctrl2components(action, joint_vel)
    }