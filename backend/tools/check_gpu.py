"""
GPU 环境检查脚本

用法:
    D:/Python/python.exe tools/check_gpu.py

功能:
    1. 打印 Python 路径和 PyTorch 版本
    2. 检测 CUDA 是否可用
    3. 如果不可用，给出重装 CUDA 版 PyTorch 的具体命令
"""

import sys
import os

# 把 backend 目录加入 path（方便从项目根目录运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title):
    """打印分隔标题"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_python():
    """检查 Python 环境"""
    print_header("Python 环境")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version:    {sys.version.split()[0]}")
    print(f"  Platform:          {sys.platform}")


def check_torch():
    """检查 PyTorch 及相关 CUDA 状态"""
    print_header("PyTorch 信息")

    try:
        import torch
        print(f"  torch.__version__:       {torch.__version__}")

        # CUDA 构建版本
        cuda_build = torch.version.cuda
        print(f"  torch.version.cuda:      {cuda_build if cuda_build else 'None （CPU-only build）'}")

        # CUDA 运行时可用性
        cuda_available = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {cuda_available}")

        if cuda_available:
            # ── CUDA 可用 ──
            print()
            print(f"  [OK] CUDA 可用！")
            print(f"  GPU 数量:                {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                print(f"  GPU[{i}]:                  {name}")

                # 显存信息
                try:
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    total_gb = total_mem / (1024 ** 3)
                    print(f"  GPU[{i}] 总显存:           {total_gb:.1f} GB")
                except Exception:
                    print(f"  GPU[{i}] 显存:             无法获取")

            # CUDA 版本
            print(f"  CUDA runtime version:    {torch.version.cuda}")
            try:
                print(f"  cuDNN version:           {torch.backends.cudnn.version()}")
            except Exception:
                print(f"  cuDNN version:           无法获取")

            print()
            print("  训练命令示例:")
            print(f"    {sys.executable} training/train.py \\")
            print(f"        --data_dir D:/python_gjw/radscout2/data/luna16 \\")
            print(f"        --epochs 20 --batch_size 8 --device cuda")

        else:
            # ── CUDA 不可用 ──
            print()
            print(f"  [NO] CUDA 不可用。")
            print()

            # 判断是否是 CPU-only build
            if "+cpu" in torch.__version__ or cuda_build is None:
                print(f"  ╔══════════════════════════════════════════════════════╗")
                print(f"  ║  检测到 CPU 版 PyTorch                               ║")
                print(f"  ║  torch.__version__ 含 '+cpu' 或 cuda build 为 None   ║")
                print(f"  ║  需要卸载 CPU 版，安装 CUDA 版。                      ║")
                print(f"  ╠══════════════════════════════════════════════════════╣")
                print(f"  ║                                                      ║")
                print(f"  ║  你的显卡:  NVIDIA RTX 4060 Laptop                   ║")
                print(f"  ║  推荐 CUDA:  12.8（或 12.6）                         ║")
                print(f"  ║                                                      ║")
                print(f"  ║  请依次执行以下命令（用当前 Python 路径）:            ║")
                print(f"  ║                                                      ║")
                print(f"  ║  1. 卸载旧版:                                        ║")
                print(f"  ║  {sys.executable} -m pip uninstall -y torch torchvision torchaudio")
                print(f"  ║                                                      ║")
                print(f"  ║  2. 安装 CUDA 12.8 版（推荐）:                       ║")
                print(f"  ║  {sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
                print(f"  ║                                                      ║")
                print(f"  ║  3. 如果 cu128 不可用，尝试 cu126:                   ║")
                print(f"  ║  {sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
                print(f"  ║                                                      ║")
                print(f"  ║  4. 安装后重新运行本脚本验证:                        ║")
                print(f"  ║  {sys.executable} tools/check_gpu.py                 ║")
                print(f"  ║                                                      ║")
                print(f"  ╚══════════════════════════════════════════════════════╝")
            else:
                print(f"  PyTorch 版本可能不是 CPU-only build，但仍然检测不到 CUDA。")
                print(f"  可能原因:")
                print(f"    1. NVIDIA 驱动版本过旧，不支持当前 CUDA 版本")
                print(f"    2. 环境变量 CUDA_VISIBLE_DEVICES 被设置为空")
                print(f"    3. PyTorch 和 CUDA 版本不匹配")
                print(f"  建议: 先运行 nvidia-smi 确认驱动正常，然后重装 PyTorch。")

    except ImportError:
        print(f"  [NO] PyTorch 未安装！")
        print(f"  请先安装 PyTorch:")
        print(f"    {sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")


def check_nvidia_driver():
    """检查 NVIDIA 驱动（通过 subprocess 调 nvidia-smi）"""
    print_header("NVIDIA 驱动")
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"  nvidia-smi 输出:")
            for line in result.stdout.strip().split("\n"):
                print(f"    {line.strip()}")
            print(f"  [OK] 驱动正常")
        else:
            print(f"  [NO] nvidia-smi 返回错误:")
            print(f"    {result.stderr.strip()}")
    except FileNotFoundError:
        print(f"  [NO] nvidia-smi 未找到")
        print(f"  请确认 NVIDIA 驱动已正确安装。")
    except Exception as e:
        print(f"  [NO] 执行 nvidia-smi 失败: {e}")


def check_other():
    """检查其他关键依赖"""
    print_header("其他关键依赖")
    deps = ["numpy", "pandas", "PIL", "SimpleITK", "sklearn"]
    for name in deps:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "OK")
            print(f"  [OK] {name}: {version}")
        except ImportError:
            print(f"  [NO] {name}: 未安装")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       RadScout v2.1 — GPU 环境检查                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    check_python()
    check_nvidia_driver()
    check_torch()
    check_other()

    print()
    print("=" * 60)
    print("  检查完成。")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
