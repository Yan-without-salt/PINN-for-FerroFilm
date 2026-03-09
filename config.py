import deepxde as dde
import numpy as np

# 随机种子与精度
dde.config.set_random_seed(2023)
dde.config.set_default_float('float32')

# 几何与时间尺度
domain_length = 4.0          # nm
time_length = 10.0            # s
L_norm = domain_length / 2
t_norm = time_length

# 材料参数（以实际为准）
a1 = -0.1725
a11 = -0.072245
a12 = 0.75255
a111 = 0.25740
a112 = 0.63036
a123 = -3.6771
c11 = 174.57
c12 = 79.278
c44 = 111.11
q11 = 10.919
q12 = 0.4485
q44 = 7.1760
G11 = 0.27689
G12 = 0.0
G44 = 0.13840
G44_ = 0.13840
gamma = 1e-3

# 边界条件类型
mb = 3
screenbot = 0.0
screentop = 0.0

# 外加电势
E_ext_max = 5.0      # V/m
f = 0.0              # Hz
omega = 2 * np.pi * f

# 各向异性介电张量
epsilon1 = 1.0
epsilon2 = 1.0
epsilon3 = 1.0
epsilon4 = 0.0

# ===== 网络结构 =====
nn_layer_size = [3] + [100] * 4 + [5]   # 输入3 (x,y,t)，输出5
activation = "tanh"
initializer = "Glorot normal"

# ===== 损失权重基础（不含观测点）=====
λ_F = [1, 1, 1, 100, 100]   # 5个PDE
loss_weights = [
    λ_F[0], λ_F[1], λ_F[2], λ_F[3], λ_F[4],   # PDE
    100, 100,  # bc_Top_traction_12, bc_Top_traction_22
    100, 100,  # bc_Bottom_disp_1, bc_Bottom_disp_2
    100,       # bc_BottomTop_charge_2
    100, 100,  # bc_BottomTop_gradient_22, bc_BottomTop_gradient_12
    100, 100,  # bc_BottomTop_gradient_11, bc_BottomTop_gradient_21
    100,
    100, 100, 100, 100, 100   # 5个周期性边界
]   # 共 5 + 12 = 17 项

# ===== 阶段定义 =====
stages = [
    {
        "name": "stage1",
        "t_start": 0,
        "t_end": 5,
        "train_adam1_iters": 30000,
        "train_adam1_lr": 1e-4,
        "train_adam2_iters": 30000,
        "train_adam2_lr": 1e-5,
        "train_lbfgs": True,
        "use_ic_obs": True,           # 是否使用观测点作为初始条件
        "ic_data_path": None,          # 第一阶段无前序，需生成初始数据
        "output_data_path": "./PINN_data/pinn_P1P2_5.npy",
        "checkpoint_dir": "./checkpoints/stage1"
    },
    {
        "name": "stage2",
        "t_start": 5,
        "t_end": 15,
        "train_adam1_iters": 10000,
        "train_adam1_lr": 1e-3,
        "train_adam2_iters": 30000,
        "train_adam2_lr": 1e-4,
        "train_lbfgs": True,
        "use_ic_obs": True,
        "ic_data_path": "./PINN_data/pinn_P1P2_5.npy",   # 使用上一阶段输出
        "output_data_path": "./PINN_data/pinn_P1P2_15.npy",
        "checkpoint_dir": "./checkpoints/stage2"
    },
    {
        "name": "stage3",
        "t_start": 15,
        "t_end": 25,
        "train_adam1_iters": 10000,
        "train_adam1_lr": 1e-3,
        "train_adam2_iters": 10000,
        "train_adam2_lr": 1e-4,
        "train_lbfgs": True,
        "use_ic_obs": True,
        "ic_data_path": "./PINN_data/pinn_P1P2_15.npy",
        "output_data_path": "./PINN_data/pinn_P1P2_25.npy",
        "checkpoint_dir": "./checkpoints/stage3"
    }
]

# 辅助函数：生成损失权重列表（根据是否使用观测点）
def get_loss_weights(use_ic_obs):
    if use_ic_obs:
        # 添加两个观测点损失权重
        return loss_weights + [100, 100]   # 共19项
    else:
        return loss_weights                # 共17项

# 用于绘图的网格
grid_x = 41
grid_y = 41
grid_t = 51
