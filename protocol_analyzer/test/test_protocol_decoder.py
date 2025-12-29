#!/usr/bin/env python3
# 测试协议解码模块

import sys
import os
# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from protocol_decoder import ProtocolDecoder
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import ARP
from scapy.layers.dns import DNS

class TestProtocolDecoder:
    """测试协议解码模块"""
    
    def setup_method(self):
        """设置测试环境"""
        self.decoder = ProtocolDecoder()
    
    def test_decode_tcp(self):
        """测试解码TCP协议"""
        # 创建TCP数据包
        packet = Ether() / IP() / TCP(dport=80, sport=12345)
        info = self.decoder.decode_packet(packet, "TCP")
        assert "TCP" in info
        assert "源端口" in info
        assert "目的端口" in info
    
    def test_decode_udp(self):
        """测试解码UDP协议"""
        # 创建UDP数据包
        packet = Ether() / IP() / UDP(dport=53, sport=12345)
        info = self.decoder.decode_packet(packet, "UDP")
        assert "UDP" in info
        assert "源端口" in info
        assert "目的端口" in info
    
    def test_decode_icmp(self):
        """测试解码ICMP协议"""
        # 创建ICMP数据包
        packet = Ether() / IP() / ICMP()
        info = self.decoder.decode_packet(packet, "ICMP")
        assert "ICMP" in info
        assert "ICMP类型" in info
    
    def test_decode_arp(self):
        """测试解码ARP协议"""
        # 创建ARP数据包
        packet = ARP()
        info = self.decoder.decode_packet(packet, "ARP")
        assert "ARP" in info
        assert "ARP操作" in info
    
    def test_decode_dns(self):
        """测试解码DNS协议"""
        # 创建DNS数据包
        packet = Ether() / IP() / UDP(dport=53) / DNS(id=1234, qr=0, qdcount=1)
        info = self.decoder.decode_packet(packet, "DNS")
        assert "DNS" in info
        assert "DNS ID" in info
    
    def test_decode_http(self):
        """测试解码HTTP协议"""
        # 创建HTTP数据包
        http_payload = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
        packet = Ether() / IP() / TCP(dport=80) / http_payload
        info = self.decoder.decode_packet(packet, "HTTP")
        assert "HTTP" in info
        assert "请求方法" in info
        assert "请求URL" in info
