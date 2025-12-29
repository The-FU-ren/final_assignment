#!/usr/bin/env python3
# 测试数据包存储模块

import sys
import os
# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from packet_storage import PacketStorage
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP

class TestPacketStorage:
    """测试数据包存储模块"""
    
    def setup_method(self):
        """设置测试环境"""
        self.storage = PacketStorage()
        self.test_file = "test_pcap.pcap"
    
    def teardown_method(self):
        """清理测试环境"""
        # 删除测试生成的文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_save_pcap(self):
        """测试保存PCAP文件"""
        # 创建测试数据包
        packets = []
        for i in range(3):
            packet = Ether() / IP() / TCP(dport=80 + i, sport=12345 + i)
            packets.append(packet)
        
        # 保存数据包
        result = self.storage.save_pcap(self.test_file, packets)
        assert result == True
        assert os.path.exists(self.test_file)
    
    def test_read_pcap(self):
        """测试读取PCAP文件"""
        # 创建测试数据包
        packets = []
        for i in range(3):
            packet = Ether() / IP() / TCP(dport=80 + i, sport=12345 + i)
            packets.append(packet)
        
        # 保存数据包
        self.storage.save_pcap(self.test_file, packets)
        
        # 读取数据包
        read_packets = self.storage.read_pcap(self.test_file)
        assert len(read_packets) == 3
        # Scapy的rdpcap返回的是Sniffed对象，不是列表
        assert hasattr(read_packets, '__iter__')
        # 确保返回的是可迭代对象
    
    def test_read_nonexistent_file(self):
        """测试读取不存在的文件"""
        # 读取不存在的文件
        read_packets = self.storage.read_pcap("nonexistent_file.pcap")
        assert isinstance(read_packets, list)
        assert len(read_packets) == 0
