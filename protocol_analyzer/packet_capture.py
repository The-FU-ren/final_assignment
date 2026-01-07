#!/usr/bin/env python3
# 数据包捕获模块

from scapy.all import sniff, get_if_list

class PacketCapture:
    def __init__(self):
        self.is_running = False
    
    def show_interfaces(self):
        """显示可用的网络接口，包含友好名称"""
        from scapy.all import IFACES
        for name, iface in IFACES.items():
            # 构建友好的接口名称，去掉设备路径
            friendly_name = f"{iface.description}"
            if iface.ip:
                friendly_name += f" - {iface.ip}"
            print(f"  - {friendly_name}")
    
    def get_if_list(self):
        """获取可用的网络接口列表"""
        return get_if_list()
    
    def get_friendly_interfaces(self):
        """获取友好的网络接口列表，返回字典{friendly_name: actual_name}"""
        from scapy.all import IFACES
        friendly_interfaces = {}
        for name, iface in IFACES.items():
            # 构建友好的接口名称，去掉设备路径
            friendly_name = f"{iface.description}"
            if iface.ip:
                friendly_name += f" - {iface.ip}"
            friendly_interfaces[friendly_name] = name
        return friendly_interfaces
    
    def capture_packets(self, interface, count=10, filter_rule=None):
        """
        从指定接口捕获数据包（固定数量）
        :param interface: 网络接口名称
        :param count: 捕获数据包数量
        :param filter_rule: BPF过滤规则
        :return: 捕获的数据包列表
        """
        try:
            # 添加超时参数，确保函数不会无限等待
            packets = sniff(
                iface=interface,
                count=count,
                filter=filter_rule if filter_rule else "",
                store=True,
                timeout=count * 2  # 设置超时时间，确保函数不会无限等待
            )
            return packets
        except Exception as e:
            # 尝试使用备用方法捕获数据包，不指定接口
            try:
                packets = sniff(
                    count=count,
                    filter=filter_rule if filter_rule else "",
                    store=True,
                    timeout=min(count * 2, 10),  # 设置最大超时时间10秒
                    lfilter=lambda pkt: True  # 确保捕获所有数据包
                )
                return packets
            except Exception:
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
            
            # 检查接口是否有效
            from scapy.all import get_if_list
            valid_interfaces = get_if_list()
            if interface not in valid_interfaces:
                raise ValueError(f"Invalid interface: {interface}")
            
            # 添加超时参数，定期检查是否需要停止
            while self.is_running:
                # 每次捕获一小段时间，然后检查是否需要停止
                sniff(
                    iface=interface,
                    filter=filter_rule if filter_rule else "",
                    prn=packet_handler,
                    store=False,
                    stop_filter=should_stop,
                    timeout=1  # 每1秒检查一次停止信号
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
                
                # 添加超时，确保线程可以响应停止信号
                while self.is_running:
                    sniff(
                        filter=filter_rule if filter_rule else "",
                        prn=packet_handler,
                        store=False,
                        stop_filter=should_stop,
                        lfilter=lambda pkt: True,  # 确保捕获所有数据包
                        timeout=1  # 每1秒检查一次停止信号
                    )
            except Exception as e2:
                print(f"使用第3层套接字实时捕获失败: {e2}")
                self.is_running = False
    
    def stop_live_capture(self):
        """
        停止实时捕获数据包
        """
        self.is_running = False
