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
    else:
        # 实时捕获数据包
        if not args.interface:
            print("请指定网络接口，使用 -i 参数")
            print("可用接口:")
            capture.show_interfaces()
            sys.exit(1)
        
        print(f"开始在接口 {args.interface} 捕获数据包...")
        print(f"过滤规则: {args.filter if args.filter else '无'}")
        print(f"捕获数量: {args.count}")
        
        packets = capture.capture_packets(args.interface, args.count, args.filter)
        
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

if __name__ == "__main__":
    main()
