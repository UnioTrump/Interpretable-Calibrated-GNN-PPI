from demo import main as train_main
from val_demo import main as val_main


def run_pipeline():
    print("Starting training...")
    train_main()
    print("Training complete.")

    print("\nStarting validation...")
    val_main()
    print("Validation complete.")
    
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    run_pipeline() 