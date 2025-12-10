#!/usr/bin/env python3
# 测试脚本，用于检查Windows系统上Scapy的接口处理方式

from scapy.all import get_if_list, IFACES, sniff

print("=== 接口信息测试 ===")

# 1. 打印所有接口列表
print("\n1. 接口列表 (get_if_list()):")
if_list = get_if_list()
for i, if_name in enumerate(if_list):
    print(f"  {i}: {if_name}")

# 2. 打印接口详细信息
print("\n2. 接口详细信息 (IFACES):")
for name, iface in IFACES.items():
    print(f"  名称: {name}")
    print(f"  描述: {iface.description}")
    print(f"  IP: {iface.ip}")
    print(f"  MAC: {iface.mac}")
    print(f"  索引: {iface.index}")
    print(f"  标志: {iface.flags}")
    print()

# 3. 尝试使用索引捕获数据包
print("\n3. 尝试使用索引捕获数据包...")
try:
    if if_list:
        # 使用第一个接口的索引
        packets = sniff(iface=0, count=2, timeout=5)
        print(f"  成功捕获 {len(packets)} 个数据包")
        for packet in packets:
            print(f"  数据包: {packet.summary()}")
except Exception as e:
    print(f"  捕获失败: {e}")

# 4. 尝试使用IFACES中的名称捕获数据包
print("\n4. 尝试使用IFACES中的名称捕获数据包...")
try:
    if IFACES:
        # 使用第一个接口的名称
        first_name = list(IFACES.keys())[0]
        packets = sniff(iface=first_name, count=2, timeout=5)
        print(f"  成功捕获 {len(packets)} 个数据包")
        for packet in packets:
            print(f"  数据包: {packet.summary()}")
except Exception as e:
    print(f"  捕获失败: {e}")
