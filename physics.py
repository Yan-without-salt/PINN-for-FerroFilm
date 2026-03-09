import deepxde as dde
import torch
import config
import numpy as np

L_norm = config.L_norm
t_norm = config.t_norm
a1 = config.a1
a11 = config.a11
a12 = config.a12
a111 = config.a111
a112 = config.a112
a123 = config.a123 
c11 = config.c11
c12 = config.c12
c44 = config.c44
q11 = config.q11
q12 = config.q12
q44 = config.q44
G11 = config.G11
G12 = config.G12
G44 = config.G44
epsilon1 = config.epsilon1
epsilon2 = config.epsilon2
epsilon4 = config.epsilon4
domain_length = config.domain_length
epsilon3 = config.epsilon3
E_ext_max = config.E_ext_max
f = config.f
G44_ = config.G44_
mb = config.mb
omega = config.omega
screenbot = config.screenbot
screentop = config.screentop
time_length = config.time_length
loss_weights = config.loss_weights
activation = config.activation
nn_layer_size = config.nn_layer_size
initializer = config.initializer


# ========== 变换函数 ==========
def transform_stage1(X, Y):
    """第一阶段输出变换（0~5s）：初始分布 + 时间依赖"""
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    u1 = Y[:, 0:1]
    u2 = Y[:, 1:2]
    phi = Y[:, 2:3]
    P1 = Y[:, 3:4]
    P2 = Y[:, 4:5]

    P1_0 = torch.cos((np.pi * x / 2) * config.L_norm) * torch.sin((np.pi * y / 4) * config.L_norm)
    P2_0 = torch.cos((np.pi * x / 2) * config.L_norm) * torch.sin((np.pi * y / 4) * config.L_norm)
    P1_new = P1 * t * config.t_norm * 0.1 + P1_0
    P2_new = P2 * t * config.t_norm * 0.1 + P2_0

    u1_new = u1 * 1e-3
    u2_new = u2 * 1e-3
    phi_new = phi * 1e-1

    return torch.cat((u1_new, u2_new, phi_new, P1_new, P2_new), dim=1)

def transform_scale_01(X, Y):
    """后续阶段输出变换（5~15s、15~25s）：简单缩放"""
    u1 = Y[:, 0:1] * 1e-3
    u2 = Y[:, 1:2] * 1e-3
    phi = Y[:, 2:3] * 1e-1
    P1 = Y[:, 3:4] * 0.1
    P2 = Y[:, 4:5] * 0.1
    return torch.cat((u1, u2, phi, P1, P2), dim=1)

def pde(X, Y):
    """
    Expresses the PDE of the phase-field model. 
    Argument X to pde(X,Y) is the input, where X[:, 0] is x-coordinate, X[:,1] is y-coordination, and X[:,2] is t(time)-coordinate.
    Argument Y to pde(X,Y) is the output, with 5 variables u1, u2, phi, P1, P2, as shown below.
    """

    u1  = Y[:, 0:1]   ## displacement in 1-direction
    u2  = Y[:, 1:2]   ## displacement in 2-direction
    phi = Y[:, 2:3]   ## electric potential 
    P1  = Y[:, 3:4]   ## polarization in 1-direction
    P2  = Y[:, 4:5]    ## polarization in 2-direction

    u1_x  = dde.grad.jacobian(Y, X, i = 0, j = 0)   ## \frac{\partial{u1}}{\partial{x}}
    u2_x  = dde.grad.jacobian(Y, X, i = 1, j = 0)   ## \frac{\partial{u2}}{\partial{x}}
    phi_x = dde.grad.jacobian(Y, X, i = 2, j = 0)   ## \frac{\partial{phi}}{\partial{x}}
    P1_x  = dde.grad.jacobian(Y, X, i = 3, j = 0)   ## \frac{\partial{P1}}{\partial{x}}
    P2_x  = dde.grad.jacobian(Y, X, i = 4, j = 0)   ## \frac{\partial{P2}}{\partial{x}}

    u1_y  = dde.grad.jacobian(Y, X, i = 0, j = 1)   ## \frac{\partial{u1}}{\partial{y}}
    u2_y  = dde.grad.jacobian(Y, X, i = 1, j = 1)   ## \frac{\partial{u2}}{\partial{y}}
    phi_y = dde.grad.jacobian(Y, X, i = 2, j = 1)   ## \frac{\partial{phi}}{\partial{y}}
    P1_y  = dde.grad.jacobian(Y, X, i = 3, j = 1)   ## \frac{\partial{P1}}{\partial{y}}
    P2_y  = dde.grad.jacobian(Y, X, i = 4, j = 1)   ## \frac{\partial{P2}}{\partial{y}}

    u1_xx   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 0)  ## \frac{\partial^2{u1}}{\partial{x}^2}
    u2_xx   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 0)  ## \frac{\partial^2{u2}}{\partial{x}^2}
    phi_xx  = dde.grad.hessian(Y, X, component= 2, i = 0, j = 0)  ## \frac{\partial^2{phi}}{\partial{x}^2}
    P1_xx   = dde.grad.hessian(Y, X, component= 3, i = 0, j = 0)  ## \frac{\partial^2{P1}}{\partial{x}^2}
    P2_xx   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 0)  ## \frac{\partial^2{P2}}{\partial{x}^2}

    u1_yy   = dde.grad.hessian(Y, X, component= 0, i = 1, j = 1)  ## \frac{\partial^2{u1}}{\partial{y}^2}
    u2_yy   = dde.grad.hessian(Y, X, component= 1, i = 1, j = 1)  ## \frac{\partial^2{u2}}{\partial{y}^2}
    phi_yy  = dde.grad.hessian(Y, X, component= 2, i = 1, j = 1)  ## \frac{\partial^2{phi}}{\partial{y}^2}
    P1_yy   = dde.grad.hessian(Y, X, component= 3, i = 1, j = 1)  ## \frac{\partial^2{P1}}{\partial{y}^2}
    P2_yy   = dde.grad.hessian(Y, X, component= 4, i = 1, j = 1)  ## \frac{\partial^2{P2}}{\partial{y}^2}

    u1_xy   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 1)  ## \frac{\partial^2{u1}}{\partial{x}\partial{y}}
    u2_xy   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 1)  ## \frac{\partial^2{u2}}{\partial{x}\partial{y}}
    phi_xy  = dde.grad.hessian(Y, X, component= 2, i = 0, j = 1)  ## \frac{\partial^2{phi}}{\partial{x}\partial{y}}
    P1_xy   = dde.grad.hessian(Y, X, component= 3, i = 0, j = 1)  ## \frac{\partial^2{P1}}{\partial{x}\partial{y}}
    P2_xy   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 1)  ## \frac{\partial^2{P2}}{\partial{x}\partial{y}}

    P1_t = dde.grad.jacobian(Y, X,  i = 3, j = 2)  ## \dot{P1}
    P2_t = dde.grad.jacobian(Y, X,  i = 4, j = 2)  ## \dot{P2}
    
    ###############################################################
    ### div(sigma) = 0 related expressions
    ### strain: plane strain assumption is used, i.e., epsilon33 = epsilon13 = epsilon23 = 0
    epsilon11_ = u1_x/L_norm   
    epsilon22_ = u2_y/L_norm
    epsilon12_ = 0.5 * (u1_y + u2_x)/L_norm

    epsilon11_x_ = u1_xx/L_norm/L_norm
    epsilon11_y_ = u1_xy/L_norm/L_norm

    epsilon12_y_ = 0.5 * (u1_yy + u2_xy)/L_norm/L_norm
    epsilon12_x_ = 0.5 * (u1_xy + u2_xx)/L_norm/L_norm

    epsilon22_x_ = u2_xy/L_norm/L_norm
    epsilon22_y_ = u2_yy/L_norm/L_norm
    
    P1_x_ = P1_x/L_norm
    P2_x_ = P2_x/L_norm
    P1_y_ = P1_y/L_norm
    P2_y_ = P2_y/L_norm
    
    P1_xx_ = P1_xx/L_norm/L_norm
    P1_yy_ = P1_yy/L_norm/L_norm
    P1_xy_ = P1_xy/L_norm/L_norm
    
    P2_xx_ = P2_xx/L_norm/L_norm
    P2_yy_ = P2_yy/L_norm/L_norm
    P2_xy_ = P2_xy/L_norm/L_norm
    
    P1_t_ = P1_t/t_norm
    P2_t_ = P2_t/t_norm

    ### stress
    sigma11 = c11 * epsilon11_ + c12 * epsilon22_ - q11 * P1 * P1 - q12 * P2 * P2
    sigma22 = c11 * epsilon22_ + c12 * epsilon11_ - q11 * P2 * P2 - q12 * P1 * P1
    sigma12 = 2 * c44 * epsilon12_ - q44 * P1 * P2
    
    ### divergence of stress
    sigma11_x = c11 * epsilon11_x_ + c12 * epsilon22_x_ - 2 * q11 * P1 * P1_x_ - 2 * q12 * P2 * P2_x_
    sigma12_y = 2 * c44 * epsilon12_y_ - q44 * P2 * P1_y_ - q44 * P1 * P2_y_
    sigma12_x = 2 * c44 * epsilon12_x_ - q44 * P2 * P1_x_ - q44 * P1 * P2_x_
    sigma22_y = c11 * epsilon22_y_ +  c12 * epsilon11_y_ - 2 * q11 * P2 * P2_y_ - 2 * q12 * P1 * P1_y_

    #=======================修改：添加各向异性介电张量计算=================
    # 电场计算 (考虑各向异性)
    E1_ = -phi_x/L_norm  # Ex
    E2_ = -phi_y/L_norm  # Ey (移除E_t，外加电场在边界条件中处理)
    
    # 电位移计算 (各向异性介电张量)
    D1 = epsilon1 * E1_ + epsilon4 * E2_ + P1
    D2 = epsilon4 * E1_ + epsilon2 * E2_ + P2
    #=======================修改结束=================
    
    #=======================修改：添加div(D) = 0的约束=================
    # divergence of electric displacement
    D1_x = epsilon1 * (-phi_xx)/L_norm/L_norm + epsilon4 * (-phi_xy)/L_norm/L_norm + P1_x_
    D2_y = epsilon4 * (-phi_xy)/L_norm/L_norm + epsilon2 * (-phi_yy)/L_norm/L_norm + P2_y_
    #=======================修改结束=================

    ###############################################################
    ### TDGL equation related expressions
    ### h_P1 = \frac{\partial{h}}{\partial{P1}}
    h_P1 = + 2 * a1 * P1 \
           + 4 * a11 * (P1**3)  \
           + 6 * a111 * (P1**5) \
           - 2 * q11 * epsilon11_ * P1 - 2 * q12 * P1 * epsilon22_ - 2 * q44 * epsilon12_ * P2 \
           - E1_ \
           + 2 * a12 * P1 * (P2**2) \
           + 4 * a112 * (P1**3) * (P2**2) + 2 * a112 * P1 * (P2**4) 
    + 2 * a123 * P1 * P2**2 

    
    ### h_P2 = \frac{\partial{h}}{\partial{P2}}
    h_P2 = + 2 * a1 * P2 \
           + 4 * a11 * (P2**3)  \
           + 6 * a111 * (P2**5) \
           - 2 * q11 * epsilon22_ * P2 - 2 * q12 * P2 * epsilon11_ - 2 * q44 * epsilon12_ * P1 \
           - E2_ \
           + 2 * a12 * P2 * (P1**2) \
           + 4 * a112 * (P2**3) * (P1**2) + 2 * a112 * P2 * (P1**4) 
    + 2 * a123 * P1 * P2**2 
    
    ### chi_{ij} = \frac{\partial{h}}{\partial{xi_{ij}}}, xi_{ij} = \frac{\partial{P_i}}{\partial{x_j}}
    chi11 = G11 * P1_x_ + G12 * P2_y_
    chi12 = G44 * (P1_y_ + P2_x_) + G44_ * (P1_y_ - P2_x_)
    chi21 = G44 * (P1_y_ + P2_x_) + G44_ * (P2_x_ - P1_y_)
    chi22 = G11 * P2_y_ + G12 * P1_x_

    ### divergence of chi_{ij}
    chi11_x = G11 * P1_xx_ + G12 * P2_xy_
    chi12_y = G44 * (P1_yy_ + P2_xy_) + G44_ * (P1_yy_ - P2_xy_)
    chi21_x = G44 * (P1_xy_ + P2_xx_) + G44_ * (P2_xx_ - P1_xy_)
    chi22_y = G11 * P2_yy_ + G12 * P1_xy_

    ### divergence of {chi_{ij}}_{2*2}
    div_P1 = chi11_x + chi12_y 
    div_P2 = chi21_x + chi22_y

    ###############################################################
    ### balance equations
    balance_mechanic_1 = sigma11_x + sigma12_y
    balance_mechanic_2 = sigma12_x + sigma22_y

    balance_electric = D1_x + D2_y

    TDGL_1 = P1_t_ + h_P1 - div_P1
    TDGL_2 = P2_t_ + h_P2 - div_P2
    
    return [balance_mechanic_1, balance_mechanic_2, balance_electric, TDGL_1, TDGL_2]

def boundary_flux(X, Y):
    """ 统一边界条件计算函数 返回9元组:
        [bc_mech_x1, bc_mech_x2, bc_mech_x12, bc_elec_x, bc_elec_y, chi11, chi22, chi12, chi21]
        - bc_mech_x1: x1方向的力学边界残差
        - bc_mech_x2: x2方向的力学边界残差
        - bc_mech_x12: x1-x2方向的力学边界残差
        - bc_elec_x: 左右边界（x=±L/2）的电学条件残差（法向为x）
        - bc_elec_y: 上下边界（y=±L/2）的电学条件残差（法向为y） """
    u1 = Y[:, 0:1]
    u2 = Y[:, 1:2]
    phi = Y[:, 2:3]
    P1 = Y[:, 3:4]
    P2 = Y[:, 4:5]
    
    # 一阶导数（归一化前）
    u1_x = dde.grad.jacobian(Y, X, i=0, j=0)
    u2_x = dde.grad.jacobian(Y, X, i=1, j=0)
    phi_x = dde.grad.jacobian(Y, X, i=2, j=0)
    P1_x = dde.grad.jacobian(Y, X, i=3, j=0)
    P2_x = dde.grad.jacobian(Y, X, i=4, j=0)
    
    u1_y = dde.grad.jacobian(Y, X, i=0, j=1)
    u2_y = dde.grad.jacobian(Y, X, i=1, j=1)
    phi_y = dde.grad.jacobian(Y, X, i=2, j=1)
    P1_y = dde.grad.jacobian(Y, X, i=3, j=1)
    P2_y = dde.grad.jacobian(Y, X, i=4, j=1)
    
 # ========== 力学量 ==========
    epsilon11_ = u1_x / L_norm
    epsilon22_ = u2_y / L_norm
    epsilon12_ = 0.5 * (u1_y + u2_x) / L_norm

    sigma11 = c11 * epsilon11_ + c12 * epsilon22_ - q11 * P1 * P1 - q12 * P2 * P2
    sigma22 = c11 * epsilon22_ + c12 * epsilon11_ - q11 * P2 * P2 - q12 * P1 * P1
    sigma12 = 2 * c44 * epsilon12_ - q44 * P1 * P2
    
    # ========== 电学量 ==========
    E1_ = -phi_x / L_norm  # Ex
    E2_ = -phi_y / L_norm  # Ey
    
    # 电位移（各向异性介电张量）
    D1 = epsilon1 * E1_ + epsilon4 * E2_ + P1  # Dx
    D2 = epsilon4 * E1_ + epsilon2 * E2_ + P2  # Dy
    
    # ========== 4. 梯度项（GPU原生）==========
    P1_x_norm = P1_x / L_norm
    P2_y_norm = P2_y / L_norm
    P1_y_norm = P1_y / L_norm
    P2_x_norm = P2_x / L_norm
    
    chi11 = G11 * P1_x_norm + G12 * P2_y_norm
    chi12 = G44 * (P1_y_norm + P2_x_norm) + G44_ * (P1_y_norm - P2_x_norm)
    chi21 = G44 * (P1_y_norm + P2_x_norm) + G44_ * (P2_x_norm - P1_y_norm)
    chi22 = G11 * P2_y_norm + G12 * P1_x_norm
    
    # ========== 5. GPU原生边界判断 ==========
    x_coord = X[:, 0]  # GPU张量
    y_coord = X[:, 1]  # GPU张量
    
    # 创建边界掩码（GPU张量）
    left_mask = torch.isclose(x_coord, x_coord.new_tensor(-1.0), atol=1e-6)
    right_mask = torch.isclose(x_coord, x_coord.new_tensor(1.0), atol=1e-6)
    bottom_mask = torch.isclose(y_coord, y_coord.new_tensor(-1.0), atol=1e-6)
    top_mask = torch.isclose(y_coord, y_coord.new_tensor(1.0), atol=1e-6)
    
    lr_mask = left_mask | right_mask  # 左右边界
    bt_mask = bottom_mask | top_mask  # 上下边界
    
    # ========== 初始化边界残差（GPU 原生）==========
    # 力学边界条件：9个分量
    # 对于薄板模型：
    # 1. 左右边界：自由边界（σ11=0, σ12=0）
    # 2. 下表面：位移固定（u1=0, u2=0）
    # 3. 上表面：应力边界（σ12=0, σ22=施加应力）
    
    # 创建残差张量
    bc_mech_11 = torch.zeros_like(sigma11)  # σ11 残差
    bc_mech_22 = torch.zeros_like(sigma22)  # σ22 残差
    bc_mech_12 = torch.zeros_like(sigma12)  # σ12 残差
    
    bc_disp_1 = torch.zeros_like(u1)  # u1 位移残差
    bc_disp_2 = torch.zeros_like(u2)  # u2 位移残差
    
    # ========== 左右边界处理：自由边界 ==========
    if lr_mask.any():
        # 左右边界：σ11 = 0, σ12 = 0
        bc_mech_11[lr_mask] = sigma11[lr_mask]  # σ11 应为 0
        bc_mech_12[lr_mask] = sigma12[lr_mask]  # σ12 应为 0
    
    # ========== 下表面处理：位移固定 ==========   if bottom_only.any():
        # 下表面 (x_3 = -h_s): u1 = 0, u2 = 0
        bc_disp_1[bottom_mask] = u1[bottom_mask]  # u1 应为 0
        bc_disp_2[bottom_mask] = u2[bottom_mask]  # u2 应为 0
    
    # ========== 上表面处理：应力边界 ==========
    if top_mask.any():
        # 上表面 (x_3 = h_f): σ12 = 0, σ22 = σ22_app
        # 假设施加的应力 σ22_app = 0（无外加法向应力）
        sigma22_app = torch.zeros_like(sigma22)
        bc_mech_12[top_mask] = sigma12[top_mask]  # σ12 应为 0
        bc_mech_22[top_mask] = sigma22[top_mask] - sigma22_app[top_mask]  # σ22 - σ22_app 应为 0

#######################################电学部分###########################################
    # ========== 6. 初始化电学边界残差（GPU原生）==========
    bc_elec_x = torch.zeros_like(D1)
    bc_elec_y = torch.zeros_like(D2)
    
    # ========== 7. 左右边界处理（GPU原生）==========
    if lr_mask.any():
        bc_elec_x[lr_mask] = D1[lr_mask]  # ✅ 直接索引赋值（支持张量）
    
    
       
    # ========== 8. 上下边界处理（GPU原生，5种边界条件）==========
    if bt_mask.any():
        # 安全地获取时间：如果 X 有第三列，则取归一化时间；否则假设时间为 0
        if X.shape[1] >= 3:
            t_phys = X[:, 2:3] * t_norm
        else:
            # 创建一个与 X[:,0] 形状相同的零张量（假设时间为 0）
            t_phys = torch.zeros_like(X[:, 0:1])

        # 根据 mb 选择边界条件
        if mb == 1:  # Dₙ=0（全绝缘）
            bc_elec_y[bt_mask] = D2[bt_mask]

        elif mb == 2:  # 固定电势（全金属电极）
            phi_cp = E_ext_max * torch.cos(omega * t_phys)
            phi_cm = torch.zeros_like(phi_cp)
            bc_elec_y[bottom_mask] = phi[bottom_mask] - phi_cp[bottom_mask]
            bc_elec_y[top_mask] = phi[top_mask] - phi_cm[top_mask]

        elif mb == 3:  # 混合：底Dₙ=0，顶固定（φ=0）
            bc_elec_y[bottom_mask] = D2[bottom_mask]
            bc_elec_y[top_mask] = phi[top_mask]

        elif mb == 4:  # 混合：底固定，顶Dₙ=0
            phi_cp = E_ext_max * torch.cos(omega * t_phys)
            bc_elec_y[bottom_mask] = phi[bottom_mask] - phi_cp[bottom_mask]
            bc_elec_y[top_mask] = D2[top_mask]

        elif mb == 5:  # 有限尺寸边界
            eps3 = x_coord.new_tensor(epsilon3)
            sb = x_coord.new_tensor(screenbot)
            st = x_coord.new_tensor(screentop)

            D2_bottom = eps3 * E2_ + (1.0 - sb) * P2
            D2_top = eps3 * E2_ + (1.0 - st) * P2

            bc_elec_y[bottom_mask] = D2_bottom[bottom_mask]
            bc_elec_y[top_mask] = D2_top[top_mask]

    # ========== 9. 返回9元组（GPU原生）==========
    return [sigma11, sigma22, sigma12, bc_disp_1, bc_disp_2, bc_elec_x, bc_elec_y, chi11, chi22, chi12, chi21]

def boundary_left_right(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm))

def boundary_top(X, on_boundary):
    """上表面 (x_3 = h_f)"""
    return on_boundary and np.isclose(X[1], domain_length/2/L_norm)

def boundary_bottom(X, on_boundary):
    """下表面 (x_3 = -h_s)"""
    return on_boundary and np.isclose(X[1], -1*domain_length/2/L_norm)

def boundary_all(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm) or np.isclose(X[1], -1*domain_length/2/L_norm) or np.isclose(X[1], domain_length/2/L_norm))


# 输出变换函数（可根据阶段选择）
def transform_scale_only(X, Y):
    """简单缩放（用于除第一阶段外的其他阶段）"""
    u1, u2, phi, P1, P2 = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4], Y[:, 4:5]
    u1_new = u1 * 1e-3
    u2_new = u2 * 1e-3
    phi_new = phi * 1e-1
    P1_new = P1 * 0.1
    P2_new = P2 * 0.1
    return torch.cat((u1_new, u2_new, phi_new, P1_new, P2_new), dim=1)

def transform_stage1(X, Y):
    """第一阶段：含初始分布硬约束"""
    x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    u1, u2, phi, P1, P2 = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4], Y[:, 4:5]
    # 初始分布
    P1_0 = torch.cos((np.pi * x / 2) * L_norm) * torch.sin((np.pi * y / 4) * L_norm)
    P2_0 = torch.cos((np.pi * x / 2) * L_norm) * torch.sin((np.pi * y / 4) * L_norm)
    # 硬约束构造
    P1_new = P1 * t * t_norm * 0.1 + P1_0
    P2_new = P2 * t * t_norm * 0.1 + P2_0
    u1_new = u1 * 1e-3
    u2_new = u2 * 1e-3
    phi_new = phi * 1e-1
    return torch.cat((u1_new, u2_new, phi_new, P1_new, P2_new), dim=1)

def build_net(stage_name="stage2"):
    net = dde.nn.FNN(nn_layer_size, activation, initializer)
    if stage_name == "stage1":
        net.apply_output_transform(transform_stage1)
    else:
        net.apply_output_transform(transform_scale_only)
    return net

def build_geomtime(t_start, t_end):
    """
    构建时空几何对象
    t_start, t_end: 物理时间（秒），函数内部会归一化
    """
    # 空间范围（归一化坐标）：domain_length/2/L_norm = 1，所以范围是 [-1, 1]
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    # 时间范围（归一化）
    t_min, t_max = t_start / t_norm, t_end / t_norm

    # 矩形空间 + 时间区间
    geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    return geomtime

def build_bc_list(geomtime, use_ic_obs, ic_data_path):
    """
    构建所有边界条件（包括观测点）
    返回一个 list，包含所有 dde.icbc.BC 对象

    """

    bc_list = []

    # ========== 1. 周期性边界（左右） ==========
    # 5 个输出变量：u1, u2, phi, P1, P2
    for comp in range(5):
        periodic_bc = dde.icbc.PeriodicBC(
        geomtime,
        component_x=0,                     # x 方向周期性
        on_boundary=lambda X, on_boundary: on_boundary and (np.isclose(X[0], -1.0) or np.isclose(X[0], 1.0)),
        component=comp                      # 输出变量分量
    )
        bc_list.append(periodic_bc)


    # ========== 2. 底部边界条件 (y = -1) ==========
    # 2.1 位移固定 u1=0, u2=0
    def bottom_u1(X, Y, X_train):
        return boundary_flux(X, Y)[3]   # bc_disp_1
    bc_bottom_u1 = dde.icbc.OperatorBC(geomtime, bottom_u1, boundary_bottom)
    bc_list.append(bc_bottom_u1)

    def bottom_u2(X, Y, X_train):
        return boundary_flux(X, Y)[4]   # bc_disp_2
    bc_bottom_u2 = dde.icbc.OperatorBC(geomtime, bottom_u2, boundary_bottom)
    bc_list.append(bc_bottom_u2)

    # 2.2 底部电学条件（由 mb 决定，已封装在 boundary_flux 的 bc_elec_y 中）
    def bottom_elec(X, Y, X_train):
        return boundary_flux(X, Y)[6]   # bc_elec_y
    bc_bottom_elec = dde.icbc.OperatorBC(geomtime, bottom_elec, boundary_bottom)
    bc_list.append(bc_bottom_elec)

    # 2.3 底部梯度条件：chi22=0, chi12=0
    def bottom_chi22(X, Y, X_train):
        return boundary_flux(X, Y)[8]   # chi22
    bc_bottom_chi22 = dde.icbc.OperatorBC(geomtime, bottom_chi22, boundary_bottom)
    bc_list.append(bc_bottom_chi22)

    def bottom_chi12(X, Y, X_train):
        return boundary_flux(X, Y)[9]   # chi12
    bc_bottom_chi12 = dde.icbc.OperatorBC(geomtime, bottom_chi12, boundary_bottom)
    bc_list.append(bc_bottom_chi12)

    # ========== 3. 顶部边界条件 (y = 1) ==========
    # 3.1 牵引力条件 sigma12=0, sigma22=0
    def top_sigma12(X, Y, X_train):
        return boundary_flux(X, Y)[2]   # sigma12
    bc_top_sigma12 = dde.icbc.OperatorBC(geomtime, top_sigma12, boundary_top)
    bc_list.append(bc_top_sigma12)

    def top_sigma22(X, Y, X_train):
        return boundary_flux(X, Y)[1]   # sigma22
    bc_top_sigma22 = dde.icbc.OperatorBC(geomtime, top_sigma22, boundary_top)
    bc_list.append(bc_top_sigma22)

    # 3.2 顶部电学条件（同样使用 bc_elec_y）
    def top_elec(X, Y, X_train):
        return boundary_flux(X, Y)[6]   # bc_elec_y
    bc_top_elec = dde.icbc.OperatorBC(geomtime, top_elec, boundary_top)
    bc_list.append(bc_top_elec)

    # 3.3 顶部梯度条件：chi22=0, chi12=0
    def top_chi22(X, Y, X_train):
        return boundary_flux(X, Y)[8]   # chi22
    bc_top_chi22 = dde.icbc.OperatorBC(geomtime, top_chi22, boundary_top)
    bc_list.append(bc_top_chi22)

    def top_chi12(X, Y, X_train):
        return boundary_flux(X, Y)[9]   # chi12
    bc_top_chi12 = dde.icbc.OperatorBC(geomtime, top_chi12, boundary_top)
    bc_list.append(bc_top_chi12)

    # ========== 4. 观测点（初始条件） ==========
    if use_ic_obs and ic_data_path is not None:
        # 读取数据文件：期望包含列 [x, y, P1, P2]
        data = np.load(ic_data_path)
        x_obs = data[:, 0]
        y_obs = data[:, 1]
        P1_obs = data[:, 2]
        P2_obs = data[:, 3]

        # 时间点 t_start（归一化后）
        # 注意：观测点数据对应 t_start 时刻，几何对象中的时间需设置为 t_start/t_norm
        # 构建时空点：需要将 (x, y, t) 组合
        # 获取归一化起始时间（几何对象的时间域起始点）
        t_start_norm = geomtime.timedomain.t0
        t_obs = np.full_like(x_obs, t_start_norm)
        X_obs = np.column_stack((x_obs, y_obs, t_obs))

        # 为 P1 和 P2 分别创建观测点约束
        # 使用 PointSetBC，其中函数返回预测值与观测值的差
        bc_p1_obs = dde.icbc.PointSetBC(X_obs, P1_obs.reshape(-1,1), component=3)  # component 3 对应 P1
        bc_p2_obs = dde.icbc.PointSetBC(X_obs, P2_obs.reshape(-1,1), component=4)  # component 4 对应 P2
        bc_list.extend([bc_p1_obs, bc_p2_obs])

    return bc_list