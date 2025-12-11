#!/usr/bin/env python3
# 主程序入口

from packet_capture import PacketCapture
from protocol_identifier import ProtocolIdentifier
from protocol_decoder import ProtocolDecoder
from packet_storage import PacketStorage
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='协议分析软件')
    parser.add_argument('-i', '--interface', type=str, help='指定网络接口')
    parser.add_argument('-c', '--count', type=int, default=10, help='捕获数据包数量')
    parser.add_argument('-f', '--filter', type=str, help='BPF过滤规则')
    parser.add_argument('-o', '--output', type=str, help='输出PCAP文件路径')
    parser.add_argument('-r', '--read', type=str, help='读取PCAP文件进行分析')
    
    args = parser.parse_args()
    
    # 初始化各个模块
    capture = PacketCapture()
    identifier = ProtocolIdentifier()
    decoder = ProtocolDecoder()
    storage = PacketStorage()
    
    if args.read:
        # 读取PCAP文件进行分析
        packets = storage.read_pcap(args.read)
        for packet in packets:
            protocol = identifier.identify_protocol(packet)
            decoded_info = decoder.decode_packet(packet, protocol)
            print("\n" + "="*50)
            print(f"协议类型: {protocol}")
            print(decoded_info)
        return 0  # 成功退出
    else:
        # 实时捕获数据包
        if not args.interface:
            print("请指定网络接口，使用 -i 参数")
            print("可用接口:")
            capture.show_interfaces()
            return 1  # 缺少必要参数，返回错误码1
        
        # 处理友好接口名称，转换为实际接口名称
        actual_interface = args.interface
        try:
            from scapy.all import IFACES
            # 检查是否直接提供了实际接口名称
            if actual_interface not in IFACES:
                # 尝试从友好名称中提取实际接口名称
                for name, iface in IFACES.items():
                    # 构建友好名称进行匹配
                    friendly_name = f"{iface.description} ({name})"
                    if iface.ip:
                        friendly_name += f" - {iface.ip}"
                    # 检查是否包含友好名称
                    if actual_interface in friendly_name:
                        actual_interface = name
                        break
        except Exception as e:
            print(f"处理接口名称时出错: {e}")
        
        print(f"开始在接口 {actual_interface} 捕获数据包...")
        print(f"过滤规则: {args.filter if args.filter else '无'}")
        print(f"捕获数量: {args.count}")
        
        packets = capture.capture_packets(actual_interface, args.count, args.filter)
        
        if args.output:
            # 保存到PCAP文件
            storage.save_pcap(args.output, packets)
            print(f"数据包已保存到: {args.output}")
        
        # 分析数据包
        for packet in packets:
            protocol = identifier.identify_protocol(packet)
            decoded_info = decoder.decode_packet(packet, protocol)
            print("\n" + "="*50)
            print(f"协议类型: {protocol}")
            print(decoded_info)
        return 0  # 成功退出

if __name__ == "__main__":
    import sys
    import os
    
    # 检测是否在调试环境中运行
    is_debug = False
    
    # 检查常见的调试器环境变量
    debug_env_vars = ['PYDEV_DEBUG', 'DEBUGPY_SESSION_ID', 'PYCHARM_HOSTED', 'VSCODE_PID']
    for var in debug_env_vars:
        if var in os.environ:
            is_debug = True
            break
    
    # 检查是否在pdb调试器中运行
    if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
        is_debug = True
    
    result = main()
    
    if is_debug:
        # 在调试环境中，直接返回结果，不调用sys.exit()
        print(f"调试模式：程序执行完成，返回结果：{result}")
    else:
        # 在正常运行环境中，使用sys.exit()退出
        sys.exit(result)
