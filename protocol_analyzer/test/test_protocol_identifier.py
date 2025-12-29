#!/usr/bin/env python3
# 测试协议识别模块

import sys
import os
# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from protocol_identifier import ProtocolIdentifier
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import ARP
from scapy.layers.dns import DNS

class TestProtocolIdentifier:
    """测试协议识别模块"""
    
    def setup_method(self):
        """设置测试环境"""
        self.identifier = ProtocolIdentifier()
    
    def test_identify_protocol_tcp(self):
        """测试识别TCP协议"""
        # 创建TCP数据包，使用非知名端口，确保被识别为TCP
        packet = Ether() / IP() / TCP(dport=12345, sport=54321)
        protocol = self.identifier.identify_protocol(packet)
        assert protocol == "TCP"
    
    def test_identify_protocol_udp(self):
        """测试识别UDP协议"""
        # 创建UDP数据包，使用非知名端口，确保被识别为UDP
        packet = Ether() / IP() / UDP(dport=12345, sport=54321)
        protocol = self.identifier.identify_protocol(packet)
        assert protocol == "UDP"
    
    def test_identify_protocol_icmp(self):
        """测试识别ICMP协议"""
        # 创建ICMP数据包
        packet = Ether() / IP() / ICMP()
        protocol = self.identifier.identify_protocol(packet)
        assert protocol == "ICMP"
    
    def test_identify_protocol_http(self):
        """测试识别HTTP协议"""
        # 创建HTTP数据包
        packet = Ether() / IP() / TCP(dport=80)
        protocol = self.identifier.identify_protocol(packet)
        assert protocol == "HTTP"
    
    def test_identify_protocol_dns(self):
        """测试识别DNS协议"""
        # 创建DNS数据包
        packet = Ether() / IP() / UDP(dport=53) / DNS()
        protocol = self.identifier.identify_protocol(packet)
        assert protocol == "DNS"
    
    def test_identify_protocol_arp(self):
        """测试识别ARP协议"""
        # 创建ARP数据包
        packet = ARP()
        protocol = self.identifier.identify_protocol(packet)
        assert protocol == "ARP"
    
    def test_identify_protocol_ipv6(self):
        """测试识别IPv6协议"""
        # 创建IPv6数据包
        packet = Ether() / IPv6()
        protocol = self.identifier.identify_protocol(packet)
        assert protocol == "IPv6"
