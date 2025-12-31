#!/usr/bin/env python3
# 测试网络接口显示效果

from packet_capture import PacketCapture

# 创建PacketCapture对象
capture = PacketCapture()

print("=== 网络接口显示测试 ===")
print("\n1. 显示可用网络接口:")
capture.show_interfaces()

print("\n2. 获取友好接口列表:")
friendly_interfaces = capture.get_friendly_interfaces()
for friendly_name, actual_name in friendly_interfaces.items():
    print(f"  - {friendly_name} -> {actual_name}")

print("\n测试完成!")