#!/usr/bin/env python3
# 数据包捕获模块

from scapy.all import sniff, get_if_list

class PacketCapture:
    def __init__(self):
        self.is_running = False
    
    def show_interfaces(self):
        """显示可用的网络接口"""
        interfaces = get_if_list()
        for iface in interfaces:
            print(f"  - {iface}")
    
    def get_if_list(self):
        """获取可用的网络接口列表"""
        return get_if_list()
    
    def capture_packets(self, interface, count=10, filter_rule=None):
        """
        从指定接口捕获数据包（固定数量）
        :param interface: 网络接口名称
        :param count: 捕获数据包数量
        :param filter_rule: BPF过滤规则
        :return: 捕获的数据包列表
        """
        try:
            packets = sniff(
                iface=interface,
                count=count,
                filter=filter_rule if filter_rule else "",
                store=True
            )
            return packets
        except Exception as e:
            # 尝试使用第3层套接字进行捕获
            from scapy.all import conf, L3socket
            print(f"尝试使用第3层套接字进行捕获: {e}")
            try:
                # 当没有WinPcap时，直接使用scapy的sniff函数，不指定iface
                # 并调整超时时间，避免无限等待
                packets = sniff(
                    count=count,
                    filter=filter_rule if filter_rule else "",
                    store=True,
                    timeout=5,  # 设置超时时间5秒
                    lfilter=lambda pkt: True  # 确保捕获所有数据包
                )
                return packets
            except Exception as e2:
                print(f"使用第3层套接字捕获失败: {e2}")
                return []
    
    def start_live_capture(self, interface, callback, filter_rule=None):
        """
        开始实时捕获数据包
        :param interface: 网络接口名称
        :param callback: 捕获到数据包时的回调函数
        :param filter_rule: BPF过滤规则
        """
        try:
            self.is_running = True
            
            def packet_handler(packet):
                callback(packet)
            
            # 使用stop_filter来检查是否需要停止捕获
            def should_stop(packet):
                return not self.is_running
            
            sniff(
                iface=interface,
                filter=filter_rule if filter_rule else "",
                prn=packet_handler,
                store=False,
                stop_filter=should_stop
            )
        except Exception as e:
            # 尝试使用第3层套接字进行实时捕获
            from scapy.all import conf
            print(f"尝试使用第3层套接字进行实时捕获: {e}")
            try:
                # 不指定iface参数，让系统自动处理
                
                def packet_handler(packet):
                    callback(packet)
                
                # 使用stop_filter来检查是否需要停止捕获
                def should_stop(packet):
                    return not self.is_running
                
                # 对于实时捕获，不设置超时，而是依赖stop_filter来停止
                sniff(
                    filter=filter_rule if filter_rule else "",
                    prn=packet_handler,
                    store=False,
                    stop_filter=should_stop,
                    lfilter=lambda pkt: True,  # 确保捕获所有数据包
                    timeout=0  # 实时捕获不设置超时
                )
            except Exception as e2:
                print(f"使用第3层套接字实时捕获失败: {e2}")
                self.is_running = False
    
    def stop_live_capture(self):
        """
        停止实时捕获数据包
        """
        self.is_running = False
