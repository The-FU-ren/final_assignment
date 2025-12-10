#!/usr/bin/env python3
# 简单的ICMP捕获测试

from packet_capture import PacketCapture
from protocol_identifier import ProtocolIdentifier

print("=== 简单ICMP捕获测试 ===")

# 1. 测试ICMP协议识别
print("\n1. 测试ICMP协议识别...")
identifier = ProtocolIdentifier()

# 2. 尝试捕获数据包（不指定接口和过滤规则）
print("\n2. 尝试捕获数据包（不指定接口，超时5秒）...")
capture = PacketCapture()

print("   尝试捕获3个数据包...")
try:
    # 不指定接口，不指定过滤规则，让系统自动处理
    packets = capture.capture_packets(
        interface="",  # 不指定接口
        count=3,
        filter_rule=""
    )
    
    if packets:
        print(f"   ✓ 成功捕获 {len(packets)} 个数据包")
        for i, packet in enumerate(packets):
            protocol = identifier.identify_protocol(packet)
            print(f"   数据包 {i+1}: 协议={protocol}")
    else:
        print("   ✗ 未捕获到数据包")
        
except Exception as e:
    print(f"   ✗ 捕获失败: {e}")

# 3. 测试GUI中打开PCAP文件的功能
print("\n3. 测试读取PCAP文件...")
from packet_storage import PacketStorage
storage = PacketStorage()

try:
    packets = storage.read_pcap("test_packets.pcap")
    print(f"   ✓ 成功读取 {len(packets)} 个数据包")
    for i, packet in enumerate(packets):
        protocol = identifier.identify_protocol(packet)
        print(f"   数据包 {i+1}: 协议={protocol}")
except Exception as e:
    print(f"   ✗ 读取失败: {e}")

print("\n=== 简单ICMP捕获测试完成 ===")
