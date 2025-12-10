#!/usr/bin/env python3
# 测试脚本，验证新添加的协议分析功能

from protocol_identifier import ProtocolIdentifier
from protocol_decoder import ProtocolDecoder
from scapy.all import IP, TCP, UDP, ICMP, ARP, DNS, DNSQR, DNSRR, Ether, Raw

print("=== 新协议分析功能测试 ===")

# 初始化模块
identifier = ProtocolIdentifier()
decoder = ProtocolDecoder()

# 创建测试数据包
print("\n=== 创建测试数据包 ===")

# 1. SSH数据包
ssh_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=22) / Raw(b"SSH-2.0-OpenSSH_8.0\n")

# 2. Telnet数据包
telnet_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=23) / Raw(b"\xff\xfd\x03\xff\xfd\x18\xff\xfd\x20\xff\xfd\x23\xff\xfd\x27")

# 3. DNS请求数据包
dns_req_packet = Ether() / IP(src="192.168.1.1", dst="8.8.8.8") / UDP(sport=12345, dport=53) / DNS(id=1, qr=0, qdcount=1, ancount=0, nscount=0, arcount=0, qd=DNSQR(qname="example.com", qtype=1, qclass=1))

# 4. DNS响应数据包
dns_resp_packet = Ether() / IP(src="8.8.8.8", dst="192.168.1.1") / UDP(sport=53, dport=12345) / DNS(id=1, qr=1, qdcount=1, ancount=1, nscount=0, arcount=0, qd=DNSQR(qname="example.com", qtype=1, qclass=1), an=DNSRR(rrname="example.com", type=1, rclass=1, ttl=3600, rdata="93.184.216.34"))

# 5. ICMP回显请求数据包
icmp_echo_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / ICMP(type=8, code=0, id=12345, seq=1)

# 6. ICMP回显应答数据包
icmp_reply_packet = Ether() / IP(src="192.168.1.2", dst="192.168.1.1") / ICMP(type=0, code=0, id=12345, seq=1)

# 7. ARP请求数据包
arp_req_packet = Ether() / ARP(op=1, hwsrc="00:11:22:33:44:55", psrc="192.168.1.1", hwdst="ff:ff:ff:ff:ff:ff", pdst="192.168.1.2")

# 8. ARP响应数据包
arp_resp_packet = Ether() / ARP(op=2, hwsrc="00:11:22:33:44:66", psrc="192.168.1.2", hwdst="00:11:22:33:44:55", pdst="192.168.1.1")

# 测试协议识别
print("\n=== 测试协议识别 ===")
test_packets = [
    ("SSH", ssh_packet),
    ("Telnet", telnet_packet),
    ("DNS请求", dns_req_packet),
    ("DNS响应", dns_resp_packet),
    ("ICMP回显请求", icmp_echo_packet),
    ("ICMP回显应答", icmp_reply_packet),
    ("ARP请求", arp_req_packet),
    ("ARP响应", arp_resp_packet)
]

for expected_proto, packet in test_packets:
    identified_proto = identifier.identify_protocol(packet)
    print(f"{expected_proto}: 预期={expected_proto}, 识别={identified_proto} {'✓' if expected_proto == identified_proto else '✗'}")

# 测试协议解码
print("\n=== 测试协议解码 ===")
for expected_proto, packet in test_packets:
    print(f"\n--- 解码 {expected_proto} 数据包 ---")
    identified_proto = identifier.identify_protocol(packet)
    decoded_info = decoder.decode_packet(packet, identified_proto)
    print(decoded_info)

print("\n=== 测试完成 ===")
