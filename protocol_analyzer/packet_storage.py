#!/usr/bin/env python3
# 数据包存储模块

from scapy.all import wrpcap, rdpcap

class PacketStorage:
    def __init__(self):
        pass
    
    def save_pcap(self, file_path, packets):
        """
        将数据包保存为PCAP文件
        :param file_path: 输出文件路径
        :param packets: 数据包列表
        :return: 是否保存成功
        """
        try:
            wrpcap(file_path, packets)
            return True
        except Exception as e:
            print(f"保存PCAP文件失败: {e}")
            return False
    
    def read_pcap(self, file_path):
        """
        从PCAP文件读取数据包
        :param file_path: 输入文件路径
        :return: 数据包列表
        """
        try:
            packets = rdpcap(file_path)
            return packets
        except Exception as e:
            print(f"读取PCAP文件失败: {e}")
            return []
