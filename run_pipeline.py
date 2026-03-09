# run_pipeline.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import numpy as np

import config
from train_stage import train_stage
from plot import plot_stage_results

def generate_initial_ic(config):
    """生成第一阶段的初始条件（t=0时刻的场）"""
    # 使用您在第一阶段硬约束中的函数生成P1,P2的初始分布
    grid_x, grid_y = config.grid_x, config.grid_y
    x = np.linspace(-config.domain_length/2/config.L_norm, config.domain_length/2/config.L_norm, grid_x)
    y = np.linspace(-config.domain_length/2/config.L_norm, config.domain_length/2/config.L_norm, grid_y)
    xx, yy = np.meshgrid(x, y)
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    # 解析初始分布（参考您原有的P1_0, P2_0）
    P1_0 = np.cos((np.pi * x_flat * config.L_norm / 2)) * np.sin((np.pi * y_flat * config.L_norm / 4))
    P2_0 = np.cos((np.pi * x_flat * config.L_norm / 2)) * np.sin((np.pi * y_flat * config.L_norm / 4))
    data = np.column_stack((x_flat, y_flat, P1_0, P2_0))
    np.save("./PINN_data/pinn_P1P2_0.npy", data)
    return "./PINN_data/pinn_P1P2_0.npy"

def main():
    # 确保数据目录存在
    Path("./PINN_data").mkdir(exist_ok=True)

    # 处理第一阶段：生成初始观测数据
    config.stages[0]["ic_data_path"] = generate_initial_ic(config)

    # 循环训练各阶段
    previous_model = None
    final_model_paths = []
    for stage in config.stages:
        print(f"\n===== 开始训练阶段: {stage['name']} =====")
        final_model, data_path = train_stage(config, stage, previous_model)
        final_model_paths.append(final_model)
        previous_model = final_model   # 下一阶段加载本阶段最终模型
        print(f"阶段完成，最终模型保存至: {final_model}")
        print(f"场数据保存至: {data_path}")

    # 绘图
    print("\n===== 开始绘图 =====")
    plot_stage_results(config, final_model_paths)

if __name__ == "__main__":
    main()
