#!/usr/bin/env python3
# 测试脚本，验证协议分析软件的各个功能模块

from packet_capture import PacketCapture
from protocol_identifier import ProtocolIdentifier
from protocol_decoder import ProtocolDecoder
from packet_storage import PacketStorage
from scapy.all import IP, TCP, UDP, Raw, Ether

print("=== 协议分析软件测试 ===")

# 1. 测试数据包捕获模块
print("\n1. 测试数据包捕获模块...")
capture = PacketCapture()
print("可用网络接口:")
capture.show_interfaces()

# 2. 测试协议识别模块
print("\n2. 测试协议识别模块...")
identifier = ProtocolIdentifier()

# 创建测试数据包
# HTTP数据包
http_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80) / Raw(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
# HTTPS数据包
https_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=443)
# FTP数据包
ftp_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=21)
# SMTP数据包
smtp_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=25)
# POP3数据包
pop3_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=110)
# IMAP数据包
imap_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=143)
# TCP数据包
tcp_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=5000)
# UDP数据包
udp_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / UDP(sport=12345, dport=53)

# 测试协议识别
print(f"HTTP数据包识别结果: {identifier.identify_protocol(http_packet)}")
print(f"HTTPS数据包识别结果: {identifier.identify_protocol(https_packet)}")
print(f"FTP数据包识别结果: {identifier.identify_protocol(ftp_packet)}")
print(f"SMTP数据包识别结果: {identifier.identify_protocol(smtp_packet)}")
print(f"POP3数据包识别结果: {identifier.identify_protocol(pop3_packet)}")
print(f"IMAP数据包识别结果: {identifier.identify_protocol(imap_packet)}")
print(f"TCP数据包识别结果: {identifier.identify_protocol(tcp_packet)}")
print(f"UDP数据包识别结果: {identifier.identify_protocol(udp_packet)}")

# 3. 测试协议解码模块
print("\n3. 测试协议解码模块...")
decoder = ProtocolDecoder()

print("\nHTTP数据包解码:")
print(decoder.decode_packet(http_packet, "HTTP"))

print("\nTCP数据包解码:")
print(decoder.decode_packet(tcp_packet, "TCP"))

print("\nUDP数据包解码:")
print(decoder.decode_packet(udp_packet, "UDP"))

# 4. 测试数据包存储模块
print("\n4. 测试数据包存储模块...")
storage = PacketStorage()

# 创建测试数据包列表
test_packets = [http_packet, tcp_packet, udp_packet]

# 保存到PCAP文件
test_file = "test_packets.pcap"
if storage.save_pcap(test_file, test_packets):
    print(f"成功保存测试数据包到 {test_file}")
    
    # 从PCAP文件读取
    read_packets = storage.read_pcap(test_file)
    print(f"从文件中读取到 {len(read_packets)} 个数据包")
    
    # 验证读取的数据包
    for i, packet in enumerate(read_packets):
        print(f"  数据包 {i+1}: {identifier.identify_protocol(packet)}")
else:
    print("保存数据包失败")

print("\n=== 测试完成 ===")
