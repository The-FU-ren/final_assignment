#!/usr/bin/env python3
# 测试集成功能

import sys
import os
# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from packet_capture import PacketCapture
from protocol_identifier import ProtocolIdentifier
from protocol_decoder import ProtocolDecoder
from packet_storage import PacketStorage
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP

class TestIntegration:
    """测试集成功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.capture = PacketCapture()
        self.identifier = ProtocolIdentifier()
        self.decoder = ProtocolDecoder()
        self.storage = PacketStorage()
        self.test_file = "test_integration.pcap"
    
    def teardown_method(self):
        """清理测试环境"""
        # 删除测试生成的文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_capture_identify(self):
        """测试捕获并识别数据包"""
        # 创建测试数据包
        packets = []
        for i in range(3):
            if i % 2 == 0:
                packet = Ether() / IP() / TCP(dport=80 + i, sport=12345 + i)
            else:
                packet = Ether() / IP() / UDP(dport=53 + i, sport=12345 + i)
            packets.append(packet)
        
        # 识别每个数据包的协议
        for packet in packets:
            protocol = self.identifier.identify_protocol(packet)
            assert protocol in ["TCP", "UDP", "HTTP", "DNS"]
    
    def test_identify_decode(self):
        """测试识别并解码数据包"""
        # 创建测试数据包
        packet = Ether() / IP() / TCP(dport=80, sport=12345)
        
        # 识别协议
        protocol = self.identifier.identify_protocol(packet)
        
        # 解码数据包
        info = self.decoder.decode_packet(packet, protocol)
        
        # 验证解码结果
        assert protocol in info
        assert "源端口" in info
        assert "目的端口" in info
    
    def test_capture_save_read(self):
        """测试捕获、保存和读取数据包"""
        # 创建测试数据包
        packets = []
        for i in range(3):
            packet = Ether() / IP() / TCP(dport=80 + i, sport=12345 + i)
            packets.append(packet)
        
        # 保存数据包
        self.storage.save_pcap(self.test_file, packets)
        
        # 读取数据包
        read_packets = self.storage.read_pcap(self.test_file)
        
        # 验证读取结果
        assert len(read_packets) == 3
        
        # 识别并解码每个读取的数据包
        for packet in read_packets:
            protocol = self.identifier.identify_protocol(packet)
            info = self.decoder.decode_packet(packet, protocol)
            assert protocol in info
    
    def test_full_pipeline(self):
        """测试完整的处理流程"""
        # 创建测试数据包
        packet = Ether() / IP() / TCP(dport=80, sport=12345)
        
        # 1. 识别协议
        protocol = self.identifier.identify_protocol(packet)
        
        # 2. 解码数据包
        info = self.decoder.decode_packet(packet, protocol)
        
        # 3. 保存数据包
        self.storage.save_pcap(self.test_file, [packet])
        assert os.path.exists(self.test_file)
        
        # 4. 读取数据包
        read_packets = self.storage.read_pcap(self.test_file)
        assert len(read_packets) == 1
        
        # 5. 再次识别和解码
        read_protocol = self.identifier.identify_protocol(read_packets[0])
        read_info = self.decoder.decode_packet(read_packets[0], read_protocol)
        assert read_protocol in read_info
