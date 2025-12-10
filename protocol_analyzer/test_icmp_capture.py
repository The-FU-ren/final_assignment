#!/usr/bin/env python3
# 测试ICMP捕获功能

from packet_capture import PacketCapture
from protocol_identifier import ProtocolIdentifier
from scapy.all import ICMP, IP, sr1

print("=== ICMP捕获测试 ===")

# 1. 创建测试ICMP数据包
print("\n1. 创建测试ICMP数据包...")
test_packet = IP(src="192.168.1.1", dst="192.168.1.2") / ICMP(type=8, code=0)
print("   测试ICMP数据包创建成功")

# 2. 测试ICMP识别
print("\n2. 测试ICMP协议识别...")
identifier = ProtocolIdentifier()
protocol = identifier.identify_protocol(test_packet)
print(f"   ICMP数据包识别结果: {protocol}")
if protocol == "ICMP":
    print("   ✓ ICMP协议识别成功")
else:
    print("   ✗ ICMP协议识别失败")

# 3. 尝试捕获ICMP数据包（使用第3层套接字）
print("\n3. 尝试捕获ICMP数据包...")
capture = PacketCapture()

# 发送一个ICMP请求，以便有数据包可捕获
print("   发送ICMP请求...")
# 使用sr1发送一个ICMP请求，超时时间2秒
sr1(IP(dst="8.8.8.8") / ICMP(), timeout=2, verbose=False)

print("   尝试捕获ICMP数据包...")
# 尝试捕获2个数据包，使用icmp过滤
try:
    packets = capture.capture_packets(
        interface="",  # 不指定接口，让系统自动处理
        count=2,
        filter_rule="icmp"
    )
    
    if packets:
        print(f"   ✓ 成功捕获 {len(packets)} 个数据包")
        for i, packet in enumerate(packets):
            protocol = identifier.identify_protocol(packet)
            print(f"   数据包 {i+1}: 协议={protocol}")
    else:
        print("   ✗ 未捕获到ICMP数据包")
        
except Exception as e:
    print(f"   ✗ 捕获失败: {e}")

print("\n=== ICMP捕获测试完成 ===")
