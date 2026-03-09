# train_stage.py
import deepxde as dde
import numpy as np
from pathlib import Path
from physics import pde, build_geomtime, build_bc_list, build_net
from config import get_loss_weights
import config

def train_stage(config, stage_info, previous_model_path=None):
    # 解包阶段信息
    t_start = stage_info["t_start"]
    t_end = stage_info["t_end"]
    use_ic_obs = stage_info["use_ic_obs"]
    ic_data_path = stage_info.get("ic_data_path")
    checkpoint_dir = Path(stage_info["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 构建几何与边界条件
    geomtime = build_geomtime(t_start, t_end)
    bc_list = build_bc_list(geomtime, use_ic_obs, ic_data_path)

    # 创建数据对象
    data = dde.data.TimePDE(
        geomtime,
        pde,
        bc_list,
        num_domain=20000,
        num_boundary=4000,
        num_test=50000,
        train_distribution='Hammersley'
    )

    # 构建网络（传入阶段名以选择正确的输出变换）
    net = build_net(stage_name=stage_info["name"])
    model = dde.Model(data, net)

    # 如果提供了前序模型，恢复权重
    if previous_model_path is not None:
        model.restore(previous_model_path)

    # 编译并训练（Adam + L-BFGS）
    loss_weights = get_loss_weights(use_ic_obs)

    # 第一阶段Adam
    model.compile("adam", lr=stage_info["train_adam1_lr"], loss='MSE', loss_weights=loss_weights)
    checkpointer = dde.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / "model"),
        verbose=1,
        save_better_only=True,
        period=1000,
        monitor='loss'
    )
    pde_resampler = dde.callbacks.PDEPointResampler(period=2000)
    model.train(iterations=stage_info["train_adam1_iters"],
                display_every=1000,
                callbacks=[checkpointer, pde_resampler])

    # 第二阶段Adam（如有）
    if stage_info.get("train_adam2_iters", 0) > 0:
        model.compile("adam", lr=stage_info["train_adam2_lr"], loss='MSE', loss_weights=loss_weights)
        model.train(iterations=stage_info["train_adam2_iters"],
                    display_every=1000,
                    callbacks=[checkpointer, pde_resampler])

    # L-BFGS优化（如有）
    if stage_info.get("train_lbfgs", False):
        dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-8, maxiter=40000)
        model.compile("L-BFGS", loss='MSE', loss_weights=loss_weights)
        model.train(display_every=1000, callbacks=[checkpointer, pde_resampler])

    # 训练结束，保存最终模型
    final_model_path = checkpoint_dir / "final_model.pt"
    model.save(str(final_model_path))

    # 生成并保存t_end时刻的场数据
    grid_x = config.grid_x
    grid_y = config.grid_y
    x = np.linspace(-config.domain_length/2/config.L_norm, config.domain_length/2/config.L_norm, grid_x)
    y = np.linspace(-config.domain_length/2/config.L_norm, config.domain_length/2/config.L_norm, grid_y)
    xx, yy = np.meshgrid(x, y)
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    t_flat = np.full_like(x_flat, t_end / config.t_norm)
    X_pred = np.column_stack((x_flat, y_flat, t_flat))
    output = model.predict(X_pred)
    P1_pred = output[:, 3]
    P2_pred = output[:, 4]
    data_out = np.column_stack((x_flat, y_flat, P1_pred, P2_pred))
    np.save(stage_info["output_data_path"], data_out)

    return str(final_model_path), stage_info["output_data_path"]