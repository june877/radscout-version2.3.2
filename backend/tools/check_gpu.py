"""Print the exact Python and PyTorch GPU environment used by RadScout."""

import sys


def main():
    print(f"Python executable: {sys.executable}")
    try:
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"torch cuda build: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            print(f"GPU memory free: {free_bytes / 1024**3:.2f} GiB")
            print(f"GPU memory total: {total_bytes / 1024**3:.2f} GiB")
        else:
            print("Current PyTorch cannot use CUDA.")
            if "+cpu" in torch.__version__:
                print("The installed PyTorch build is CPU-only.")
    except Exception as exc:
        print(f"PyTorch check failed: {exc}")


if __name__ == "__main__":
    main()
