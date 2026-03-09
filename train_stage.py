# train_stage.py
import deepxde as dde
import numpy as np
from pathlib import Path
from physics import pde, build_geomtime, build_bc_list, build_net
from config import get_loss_weights
import config

class LossMonitor(dde.callbacks.Callback):
    def __init__(self, log_path, patience=1000, min_delta=1e-6, verbose=1):
        super().__init__()
        self.log_path = log_path
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = np.inf
        self.counter = 0
        # 打开log文件，准备写入
        self.log_file = open(log_path, 'a')
        # 写入表头（如果文件为空）
        if self.log_file.tell() == 0:
            self.log_file.write("epoch\ttrain_loss\ttest_loss\n")
            self.log_file.flush()

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        loss_train = self.model.train_state.loss_train
        loss_test = self.model.train_state.loss_test
        loss_weights = self.model.loss_weights

        # 计算加权总损失
        train_total = sum(w * l for w, l in zip(loss_weights, loss_train))
        test_total = sum(w * l for w, l in zip(loss_weights, loss_test)) if loss_test else None

        # 每1000步记录
        if epoch % 1000 == 0:
            log_line = f"{epoch}\t{train_total:.6e}"
            if test_total is not None:
                log_line += f"\t{test_total:.6e}"
            else:
                log_line += "\tNone"
            self.log_file.write(log_line + "\n")
            self.log_file.flush()
            if self.verbose:
                print(f"[LossMonitor] Epoch {epoch}: train_loss = {train_total:.6e}, test_loss = {test_total if test_total is None else f'{test_total:.6e}'}")

        # 检查是否改善
        if train_total < self.best_loss - self.min_delta:
            self.best_loss = train_total
            self.counter = 0
        else:
            self.counter += 1

        # 如果连续patience步未改善，停止训练
        if self.counter >= self.patience:
            if self.verbose:
                print(f"[LossMonitor] Early stopping triggered after {epoch} epochs (no improvement for {self.patience} steps).")
            raise dde.callbacks.StopTraining

    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()

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

    # 获取损失权重（用于编译，同时也用于回调中计算加权损失，但回调会从模型获取）
    loss_weights = get_loss_weights(use_ic_obs)

    # 定义log文件路径
    log_path = checkpoint_dir / "training_log.txt"

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
    loss_monitor = LossMonitor(log_path, patience=1000, min_delta=1e-6, verbose=1)
    model.train(iterations=stage_info["train_adam1_iters"],
                display_every=1000,
                callbacks=[checkpointer, pde_resampler, loss_monitor])

    # 第二阶段Adam（如有）
    if stage_info.get("train_adam2_iters", 0) > 0:
        model.compile("adam", lr=stage_info["train_adam2_lr"], loss='MSE', loss_weights=loss_weights)
        # 重新创建监控器，重置状态
        loss_monitor = LossMonitor(log_path, patience=1000, min_delta=1e-6, verbose=1)
        model.train(iterations=stage_info["train_adam2_iters"],
                    display_every=1000,
                    callbacks=[checkpointer, pde_resampler, loss_monitor])

    # L-BFGS优化（如有）
    if stage_info.get("train_lbfgs", False):
        dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-8, maxiter=40000)
        model.compile("L-BFGS", loss='MSE', loss_weights=loss_weights)
        loss_monitor = LossMonitor(log_path, patience=1000, min_delta=1e-6, verbose=1)
        model.train(display_every=1000, callbacks=[checkpointer, pde_resampler, loss_monitor])

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