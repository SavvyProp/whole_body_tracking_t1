import torch
from isaaclab.utils.math import quat_apply

bodies = ['Trunk', 'H1', 'AL1', 'AR1', 'Waist', 'H2', 'AL2', 'AR2', 'Hip_Pitch_Left', 'Hip_Pitch_Right', 'AL3', 'AR3', 'Hip_Roll_Left', 'Hip_Roll_Right', 'left_hand_link', 'right_hand_link', 'Hip_Yaw_Left', 'Hip_Yaw_Right', 'Shank_Left', 'Shank_Right', 'Ankle_Cross_Left', 'Ankle_Cross_Right', 'left_foot_link', 'right_foot_link']
joints = ['AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_pitch', 'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Elbow_Yaw', 'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'Left_Knee_Pitch', 'Right_Knee_Pitch', 'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 'Left_Ankle_Roll', 'Right_Ankle_Roll']

CTRL_NUM = 23
EEF_NUM = 4

TORQUE_LIMITS = torch.tensor([
    7, 18, 18, 30, 7, 18, 18, 45, 45, 18, 18, 30, 30, 18, 18, 30, 30, 60, 60, 20, 20, 15, 15
])

MASS = 31.614357
ANGULAR_INERTIA = torch.tensor(
    [[ 2.77498525e+00,  5.36123413e-04,  2.12637797e-01],
 [ 5.36123413e-04,  2.64427940e+00, -2.98730940e-03],
 [ 2.12637797e-01, -2.98730940e-03,  4.91490757e-01]])
INV_ANGULAR_INERTIA = torch.linalg.inv(ANGULAR_INERTIA)

EEF_BODIES = ["left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"]

EEF_IDS = [bodies.index(name) for name in EEF_BODIES]

def ctrl2logits(act):
    des_pos = act[..., 0:CTRL_NUM]
    des_com_vel = act[..., CTRL_NUM:CTRL_NUM + 3]
    des_com_angvel = act[..., CTRL_NUM + 3 : CTRL_NUM + 6]
    w = act[..., CTRL_NUM + 6 : CTRL_NUM + EEF_NUM + 6]
    torque = act[..., CTRL_NUM + EEF_NUM + 6:
              CTRL_NUM * 2 + EEF_NUM + 6]
    d_gain = act[..., -2:]
    logits = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "w": w,
        "torque": torque,
        "d_gain": d_gain
    }
    return logits

def ctrl2components(act, qvel):
    logits = ctrl2logits(act)
    des_pos = logits["des_pos"]

    logits["des_com_angvel"] *= 0.20
    des_angvel_mag = torch.norm(logits["des_com_angvel"], dim =-1, keepdim=True)
    des_angvel_mag_clipped = torch.clamp(des_angvel_mag, max = 2.0)
    des_angvel = logits["des_com_angvel"] * (des_angvel_mag_clipped / (1e-6 + des_angvel_mag))
    
    logits["des_com_vel"] *= 0.05
    des_vel_mag = torch.norm(logits["des_com_vel"], dim =-1, keepdim=True)
    des_vel_mag_clipped = torch.clamp(des_vel_mag, max = 3.0)
    des_vel = logits["des_com_vel"] * (des_vel_mag_clipped / (1e-6 + des_vel_mag))

    w = logits["w"]

    torque_logit = torch.tanh(logits["torque"])
    torque_limits = TORQUE_LIMITS.to(torque_logit.device)
    joint_vel = qvel[CTRL_NUM][..., 6:]
    tau_naive = torque_limits[None, :] * torque_logit
    spd_fac = torch.clip(torch.abs(joint_vel), min = 0.0, max = 10.0) / 10.0
    sign = torch.where(joint_vel * torque_logit >= 0, 1.0, 0.0)
    tau = tau_naive * (1.0 - spd_fac[None, :] * sign)

    d_gain_lin = torch.tanh(logits["d_gain"][:, 0]) * 3.0 + 4.0
    #d_gain_lin = jnp.tanh(logits["d_gain"][0]) * 6.0 + 7.0
    d_gain_angvel = torch.tanh(logits["d_gain"][:, 1]) * 0.05 + 0.07

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
    batch_invI = INV_ANGULAR_INERTIA.to(r.device, r.dtype).expand(r.shape[0], -1, -1)  # (N,3,3)
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
    g = torch.tensor([0, 0, -9.81, 0, 0, 0], device=eefpos.device, dtype=eefpos.dtype)  # (6,)
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
    jacs:    (N, 6*EEF_NUM, 6+CTRL_NUM)  stacked 6D Jacobians per EEF
    tau_ref: (N, CTRL_NUM)               reference joint torques

    Returns:
      big_q:   (N, 6*EEF_NUM, 6*EEF_NUM) = (J_j^T @ J_j)
      small_q: (N, 6*EEF_NUM)            = (J_j^T @ tau_ref)
    where J_j = -J[:, :, 6:]  (exclude the 6 base dofs)
    """
    # Ensure dtype/device alignment and batch dim
    tau_ref = tau_ref.to(jacs.device, jacs.dtype)
    if tau_ref.dim() == 1:
        tau_ref = tau_ref.unsqueeze(0)

    # J_j: (N, 6*EEF_NUM, CTRL_NUM)
    J_j = -jacs[..., :, 6:]
    # mat = J_j^T: (N, CTRL_NUM, 6*EEF_NUM)
    mat = J_j.transpose(-1, -2)

    # big_q = (J_j^T @ J_j): (N, 6*EEF_NUM, 6*EEF_NUM)
    big_q = torch.matmul(mat.transpose(-1, -2), mat)

    # small_q = (J_j^T @ tau_ref): (N, 6*EEF_NUM)
    small_q = torch.matmul(mat.transpose(-1, -2), tau_ref.unsqueeze(-1)).squeeze(-1)

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

    # Solve KKT @ sol_all = rhs
    try:
        sol_all = torch.linalg.solve(KKT, rhs.unsqueeze(-1)).squeeze(-1)  # (..., F+M)
    except RuntimeError:
        # Fallback to least-squares if singular
        sol_all = torch.linalg.lstsq(KKT, rhs.unsqueeze(-1)).solution.squeeze(-1)

    sol = sol_all[..., :F]                                   # (..., F)

    if squeeze_out:
        sol = sol.squeeze(0)
    return sol

def ft_ref(
    eefpos, com_pos, jacs, tau_ref, com_ref, w
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

    tau = -jacs[..., :, 6:].transpose(-1, -2) @ f[..., None]
    return tau.squeeze(-1)

def highlvlPD(qpos, qvel, 
              lin_gain, angvel_gain,
              des_vel, des_angvel,
              com_vel):
    q_wb = qpos[:, 3:7]
    global_des_vel = quat_apply(q_wb, des_vel)

    com_acc = lin_gain[:, None] * (global_des_vel - com_vel)

    com_angvel = qvel[:, 3:6]
    ang_acc = angvel_gain[:, None] * (des_angvel - com_angvel)
    return com_acc, ang_acc

def step(com_pos, com_vel,
         jacs,
         eefpos,
         qpos, qvel,
         action):
    comp_dict = ctrl2components(action, qvel)
    com_acc, ang_acc = highlvlPD(
        qpos, qvel,
        comp_dict["d_gain_lin"], comp_dict["d_gain_angvel"],
        comp_dict["des_com_vel"], comp_dict["des_com_angvel"],
        com_vel
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
    torque_limits = TORQUE_LIMITS.to(tau.device, tau.dtype)
    tau = torch.clamp(tau, min=-torque_limits[None, :], max=torque_limits[None, :])
    return comp_dict["des_pos"], tau

def ft_rew_info(qvel, action):
    return {
        "logits": ctrl2logits(action),
        "components": ctrl2components(action, qvel)
    }