import numpy as np
import os

# 查看最佳超参数
print("=== 最佳超参数 ===")
try:
    params = np.load('结果/数据/最佳超参数.npy', allow_pickle=True).item()
    for k, v in params.items():
        print(f"{k}: {v} ({type(v).__name__})")
except Exception as e:
    print(f"读取最佳超参数失败: {e}")

# 查看中断训练记录
print("\n=== 中断训练记录 ===")
try:
    interrupted_record = np.load('结果/数据/中断训练记录.npy', allow_pickle=True)
    print(f"记录类型: {type(interrupted_record).__name__}")
    if hasattr(interrupted_record, 'shape'):
        print(f"记录形状: {interrupted_record.shape}")
    print(f"记录内容: {interrupted_record}")
except Exception as e:
    print(f"读取中断训练记录失败: {e}")

# 查看最新的运行目录
print("\n=== 最新运行目录内容 ===")
run_dirs = [d for d in os.listdir('结果/数据') if d.startswith('run_')]
if run_dirs:
    latest_run = sorted(run_dirs)[-1]
    latest_run_path = os.path.join('结果/数据', latest_run)
    print(f"最新运行目录: {latest_run_path}")
    try:
        files = os.listdir(latest_run_path)
        if files:
            print("目录文件:")
            for f in files:
                f_path = os.path.join(latest_run_path, f)
                f_size = os.path.getsize(f_path) / 1024  # KB
                print(f"  {f} ({f_size:.2f} KB)")
        else:
            print("目录为空")
    except Exception as e:
        print(f"读取目录内容失败: {e}")
else:
    print("没有找到运行目录")

# 查看图表目录
print("\n=== 图表目录 ===")
try:
    chart_dirs = [d for d in os.listdir('结果/图表')]
    if chart_dirs:
        print("图表类型:")
        for d in chart_dirs:
            chart_path = os.path.join('结果/图表', d)
            run_chart_dirs = [dc for dc in os.listdir(chart_path) if dc.startswith('run_')]
            if run_chart_dirs:
                latest_chart = sorted(run_chart_dirs)[-1]
                latest_chart_path = os.path.join(chart_path, latest_chart)
                chart_files = os.listdir(latest_chart_path)
                print(f"  {d}: {len(chart_files)} 个文件")
except Exception as e:
    print(f"读取图表目录失败: {e}")
