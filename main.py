import argparse
from types import SimpleNamespace

# 直接从脚本中导入主功能函数
from demo import main as train_main
from val_demo import main as val_main


def run_pipeline():
    """
    一个更简洁的pipeline，用于运行训练和验证。
    它直接调用其他脚本的main函数，而不是使用子进程。
    """
    parser = argparse.ArgumentParser(
        description="Run a simplified training and validation pipeline for PPI prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 在help信息中显示默认值
    )

    # --- 关键路径参数 ---
    parser.add_argument('--data_path_train', type=str, required=True, help='Path to the training dataset .pkl file.')
    parser.add_argument('--data_path_val', type=str, required=True, help='Path to the validation dataset .pkl file.')
    parser.add_argument('--model_dir', type=str, default='./saved_models', help='Directory to save/load trained models.')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='Directory to save training plots.')

    # --- 训练超参数 ---
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--pos_weight', type=float, default=3.0, help='Positive class weight for loss function.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Model dropout rate.')
    
    # --- 训练控制 ---
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # --- 步骤 1: 运行训练 ---
    print("\n" + "="*50)
    print(" " * 18 + "STARTING TRAINING")
    print("="*50)
    
    # 为 train_main 创建一个符合其期望的参数对象
    # demo.py 内部的参数名为 `data_path`
    train_args = SimpleNamespace(
        data_path=args.data_path_train,
        model_dir=args.model_dir,
        plot_dir=args.plot_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pos_weight=args.pos_weight,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed
    )
    train_main(train_args)
    
    print("\n" + "="*50)
    print(" " * 16 + "TRAINING COMPLETED")
    print("="*50 + "\n")

    # --- 步骤 2: 运行验证 ---
    print("\n" + "="*50)
    print(" " * 17 + "STARTING VALIDATION")
    print("="*50)
    
    # 为 val_main 创建一个符合其期望的参数对象
    # val_demo.py 内部的参数名为 `full_data_path`
    val_args = SimpleNamespace(
        full_data_path=args.data_path_val,
        model_dir=args.model_dir,
        seed=args.seed,
        dropout=args.dropout,
    )
    val_main(val_args)

    print("\n" + "="*50)
    print(" " * 15 + "VALIDATION COMPLETED")
    print("="*50 + "\n")
    
    print("Pipeline finished successfully.")


if __name__ == "__main__":
    run_pipeline() 

'''
proteins are selected according to the criteria of resolution <3.0 Å and sequence homology <25%;
proteins with more than 25% sequence similarity and 90% of overlapping proteins are removed.
'''