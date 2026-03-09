# plot.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
import deepxde as dde
import torch

# 导入配置和网络构建函数
import config
from physics import build_net, L_norm, t_norm, domain_length

def plot_stage_results(config, model_paths, output_dir="./PINN_plots"):
    """
    绘制各阶段结束时刻的极化场图像

    参数:
        config: 配置模块（包含 stages 列表等）
        model_paths: list，各阶段最终模型的路径（按阶段顺序）
        output_dir: 图像保存目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 网格参数
    grid_x = config.grid_x
    grid_y = config.grid_y
    # 归一化坐标网格
    x_norm = np.linspace(-domain_length/2/L_norm, domain_length/2/L_norm, grid_x)
    y_norm = np.linspace(-domain_length/2/L_norm, domain_length/2/L_norm, grid_y)
    xx_norm, yy_norm = np.meshgrid(x_norm, y_norm)
    x_flat = xx_norm.flatten()
    y_flat = yy_norm.flatten()

    # 全局颜色范围（可选）
    global_P_min = float('inf')
    global_P_max = float('-inf')
    P_all = []

    # 首先计算全局极化强度范围，以便统一色标（可选）
    print("计算全局极化强度范围...")
    for stage_idx, model_path in enumerate(model_paths):
        # 构建网络并加载模型
        net = build_net(stage_name=config.stages[stage_idx]["name"])
        model = dde.Model(None, net)
        model.restore(model_path)

        # 当前阶段结束时间
        t_end_phys = config.stages[stage_idx]["t_end"]
        t_normed = np.full_like(x_flat, t_end_phys / t_norm)
        X_pred = np.column_stack((x_flat, y_flat, t_normed))

        output = model.predict(X_pred)
        P1 = output[:, 3]
        P2 = output[:, 4]
        P_mag = np.sqrt(P1**2 + P2**2)
        P_all.append(P_mag)
        global_P_min = min(global_P_min, P_mag.min())
        global_P_max = max(global_P_max, P_mag.max())
    print(f"全局极化强度范围: {global_P_min:.4f} ~ {global_P_max:.4f}")

    # 绘图函数
    def plot_polarization_field(P1, P2, X_phys, Y_phys, t_phys, save_path):
        """绘制单个时刻的极化场图"""
        P_mag = np.sqrt(P1**2 + P2**2)
        P_angle = np.arctan2(P2, P1)

        # 重塑为网格
        P_mag_grid = P_mag.reshape(grid_y, grid_x)
        P_angle_grid = P_angle.reshape(grid_y, grid_x)

        fig, ax = plt.subplots(figsize=(10, 8))

        # 背景颜色图
        norm = Normalize(vmin=global_P_min, vmax=global_P_max)  # 使用全局范围
        cmap = plt.cm.rainbow
        im = ax.imshow(P_mag_grid,
                       extent=[X_phys.min(), X_phys.max(), Y_phys.min(), Y_phys.max()],
                       origin='lower', cmap=cmap, norm=norm, alpha=0.8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'Polarization Magnitude $|\mathbf{P}|$ (C/m²)', fontsize=14)

        # 箭头长度函数（与您原代码一致）
        def arrow_length_function(p_mag, p_min, p_max):
            p_norm = (p_mag - p_min) / (p_max - p_min + 1e-10)
            l_min = 0.05
            l_max = 0.3
            arrow_length = l_min + (l_max - l_min) * (np.arctan(2*np.pi*0.2*(p_norm-0.5))/np.pi + 0.5)
            return arrow_length

        p_min_local = P_mag.min()
        p_max_local = P_mag.max()

        # 绘制箭头
        step = 2
        for i in range(0, grid_y, step):
            for j in range(0, grid_x, step):
                x_start = X_phys[i, j]
                y_start = Y_phys[i, j]
                arrow_len = arrow_length_function(P_mag_grid[i, j], p_min_local, p_max_local)
                x_end = x_start + arrow_len * np.cos(P_angle_grid[i, j])
                y_end = y_start + arrow_len * np.sin(P_angle_grid[i, j])

                arrow = FancyArrowPatch(
                    (x_start, y_start), (x_end, y_end),
                    arrowstyle='-|>', color='white', linewidth=2,
                    mutation_scale=5, alpha=0.95, zorder=5
                )
                ax.add_patch(arrow)

        ax.set_xlabel('x (nm)', fontsize=14)
        ax.set_ylabel('y (nm)', fontsize=14)
        ax.set_title(f'Polarization Field at t={t_phys:.2f}s\n'
                     f'(Color: |P| magnitude, Arrows: P direction)', fontsize=16)
        ax.set_aspect('equal')
        ax.grid(False)

        # 添加信息框
        ax.text(0.02, 0.98,
                f'Time: t={t_phys:.2f}s\nE_ext: {config.E_ext_max}sin(2π{config.f}t) V/m',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, zorder=100)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  图像已保存: {save_path}")

    def plot_components(P1, P2, X_phys, Y_phys, t_phys, save_path):
        """绘制两个极化分量的并排图"""
        P1_grid = P1.reshape(grid_y, grid_x)
        P2_grid = P2.reshape(grid_y, grid_x)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # P1 分量
        im1 = axes[0].imshow(P1_grid,
                              extent=[X_phys.min(), X_phys.max(), Y_phys.min(), Y_phys.max()],
                              origin='lower', cmap='RdBu_r', alpha=0.8)
        axes[0].set_title(r'$P_1$ Component', fontsize=14)
        axes[0].set_xlabel('x (nm)', fontsize=12)
        axes[0].set_ylabel('y (nm)', fontsize=12)
        fig.colorbar(im1, ax=axes[0])

        # P2 分量
        im2 = axes[1].imshow(P2_grid,
                              extent=[X_phys.min(), X_phys.max(), Y_phys.min(), Y_phys.max()],
                              origin='lower', cmap='RdBu_r', alpha=0.8)
        axes[1].set_title(r'$P_2$ Component', fontsize=14)
        axes[1].set_xlabel('x (nm)', fontsize=12)
        axes[1].set_ylabel('y (nm)', fontsize=12)
        fig.colorbar(im2, ax=axes[1])

        plt.suptitle(f'Polarization Components at t={t_phys:.2f}s', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  分量图已保存: {save_path}")

    # 为每个阶段生成图像
    for stage_idx, model_path in enumerate(model_paths):
        stage_info = config.stages[stage_idx]
        t_end_phys = stage_info["t_end"]
        print(f"\n生成阶段 {stage_info['name']} 结束时刻 t={t_end_phys:.2f}s 的图像...")

        # 加载模型
        net = build_net(stage_name=stage_info["name"])
        model = dde.Model(None, net)
        model.restore(model_path)

        # 准备输入
        t_normed = np.full_like(x_flat, t_end_phys / t_norm)
        X_pred = np.column_stack((x_flat, y_flat, t_normed))
        output = model.predict(X_pred)

        P1 = output[:, 3]
        P2 = output[:, 4]
        # 物理坐标
        X_phys = x_flat * L_norm
        Y_phys = y_flat * L_norm
        X_grid = X_phys.reshape(grid_y, grid_x)
        Y_grid = Y_phys.reshape(grid_y, grid_x)

        # 保存极化场图
        field_path = output_dir / f"polarization_field_{stage_info['name']}_t{t_end_phys:.0f}.jpg"
        plot_polarization_field(P1, P2, X_grid, Y_grid, t_end_phys, field_path)

        # 保存分量图
        comp_path = output_dir / f"polarization_components_{stage_info['name']}_t{t_end_phys:.0f}.jpg"
        plot_components(P1, P2, X_grid, Y_grid, t_end_phys, comp_path)

    print("\n所有阶段图像生成完成！")


def plot_time_series(config, model_path, time_list, output_dir="./PINN_plots_time_series"):
    """
    为单个模型生成一系列时间点的极化场图像（可选）

    参数:
        config: 配置模块
        model_path: 模型路径
        time_list: 物理时间列表（例如 [0, 1, 2, 3, 4, 5]）
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 网格参数
    grid_x = config.grid_x
    grid_y = config.grid_y
    x_norm = np.linspace(-domain_length/2/L_norm, domain_length/2/L_norm, grid_x)
    y_norm = np.linspace(-domain_length/2/L_norm, domain_length/2/L_norm, grid_y)
    xx_norm, yy_norm = np.meshgrid(x_norm, y_norm)
    x_flat = xx_norm.flatten()
    y_flat = yy_norm.flatten()

    # 加载模型（需要知道阶段名称，这里假设传入的模型路径对应某个阶段，可手动指定阶段名或从 config 推断）
    # 简化：假定第一阶段模型
    net = build_net(stage_name="stage1")  # 根据实际情况可能需要修改
    model = dde.Model(None, net)
    model.restore(model_path)

    for t_phys in time_list:
        t_normed = np.full_like(x_flat, t_phys / t_norm)
        X_pred = np.column_stack((x_flat, y_flat, t_normed))
        output = model.predict(X_pred)
        P1 = output[:, 3]
        P2 = output[:, 4]

        X_phys = x_flat * L_norm
        Y_phys = y_flat * L_norm
        X_grid = X_phys.reshape(grid_y, grid_x)
        Y_grid = Y_phys.reshape(grid_y, grid_x)

        # 绘图（复用内部函数，这里简化，直接调用 plot_polarization_field 但需要传入 P_mag 等）
        P_mag = np.sqrt(P1**2 + P2**2)
        P_angle = np.arctan2(P2, P1)

        P_mag_grid = P_mag.reshape(grid_y, grid_x)
        P_angle_grid = P_angle.reshape(grid_y, grid_x)

        # 使用全局范围（可选）
        global_P_min = P_mag.min()  # 简单处理，也可以用所有时刻的全局范围
        global_P_max = P_mag.max()

        fig, ax = plt.subplots(figsize=(10, 8))
        norm = Normalize(vmin=global_P_min, vmax=global_P_max)
        cmap = plt.cm.rainbow
        im = ax.imshow(P_mag_grid,
                       extent=[X_grid.min(), X_grid.max(), Y_grid.min(), Y_grid.max()],
                       origin='lower', cmap=cmap, norm=norm, alpha=0.8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'Polarization Magnitude $|\mathbf{P}|$ (C/m²)', fontsize=14)

        def arrow_length_function(p_mag, p_min, p_max):
            p_norm = (p_mag - p_min) / (p_max - p_min + 1e-10)
            l_min = 0.05
            l_max = 0.3
            arrow_length = l_min + (l_max - l_min) * (np.arctan(2*np.pi*0.2*(p_norm-0.5))/np.pi + 0.5)
            return arrow_length

        p_min_local = P_mag_grid.min()
        p_max_local = P_mag_grid.max()
        step = 2
        for i in range(0, grid_y, step):
            for j in range(0, grid_x, step):
                x_start = X_grid[i, j]
                y_start = Y_grid[i, j]
                arrow_len = arrow_length_function(P_mag_grid[i, j], p_min_local, p_max_local)
                x_end = x_start + arrow_len * np.cos(P_angle_grid[i, j])
                y_end = y_start + arrow_len * np.sin(P_angle_grid[i, j])

                arrow = FancyArrowPatch(
                    (x_start, y_start), (x_end, y_end),
                    arrowstyle='-|>', color='white', linewidth=2,
                    mutation_scale=5, alpha=0.95, zorder=5
                )
                ax.add_patch(arrow)

        ax.set_xlabel('x (nm)', fontsize=14)
        ax.set_ylabel('y (nm)', fontsize=14)
        ax.set_title(f'Polarization Field at t={t_phys:.2f}s', fontsize=16)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.text(0.02, 0.98,
                f'Time: t={t_phys:.2f}s\nE_ext: {config.E_ext_max}sin(2π{config.f}t) V/m',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, zorder=100)

        save_path = output_dir / f"polarization_field_t_{t_phys:.2f}.jpg"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  已保存: {save_path}")

    print("时间序列图像生成完成！")


if __name__ == "__main__":
    # 独立测试时使用
    import sys
    # 假设模型路径列表从命令行参数获取或手动指定
    # 这里仅作为示例，实际运行时通过主控脚本调用
    pass

