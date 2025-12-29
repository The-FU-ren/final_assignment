#!/usr/bin/env python3
# 测试数据包捕获模块

import sys
import os
# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from packet_capture import PacketCapture

class TestPacketCapture:
    """测试数据包捕获模块"""
    
    def test_get_friendly_interfaces(self):
        """测试获取友好接口列表"""
        capture = PacketCapture()
        interfaces = capture.get_friendly_interfaces()
        assert isinstance(interfaces, dict)
        # 确保返回的是字典类型
    
    def test_get_if_list(self):
        """测试获取接口列表"""
        capture = PacketCapture()
        interfaces = capture.get_if_list()
        assert isinstance(interfaces, list)
        # 确保返回的是列表类型
    
    def test_capture_packets(self):
        """测试捕获数据包功能"""
        capture = PacketCapture()
        # 使用超时参数，确保测试能在合理时间内完成
        packets = capture.capture_packets(interface=None, count=3, filter_rule="ip")
        # Scapy的sniff函数返回的是Sniffed对象，不是列表
        assert hasattr(packets, '__iter__')
        # 确保返回的是可迭代对象
