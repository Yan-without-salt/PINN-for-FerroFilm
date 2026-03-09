project/
├── config.py               # 统一配置文件（参数、阶段列表等）
├──physics.py             # PDE定义、边界通量、几何构造、网络构建（共享函数）
├── train_stage.py           # 单阶段训练函数（加载模型、训练、保存结果）
├── plot.py                  # 绘图函数（加载模型、生成图像）
├── run_pipeline.py          # 主控脚本（顺序执行各阶段训练 + 最终绘图）
├── PINN_data/               # 存放各阶段初始/最终场数据（自动生成）
├── checkpoints/             # 各阶段模型检查点（自动保存）
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
└── PINN_plots/              # 最终图像输出目录