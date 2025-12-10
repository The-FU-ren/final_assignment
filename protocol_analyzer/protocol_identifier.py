#!/usr/bin/env python3
# 协议识别模块

class ProtocolIdentifier:
    def __init__(self):
        pass
    
    def identify_protocol(self, packet):
        """
        识别数据包的协议类型
        :param packet: 数据包对象
        :return: 协议名称
        """
        # 检查应用层协议
        if self._is_http(packet):
            return "HTTP"
        elif self._is_https(packet):
            return "HTTPS"
        elif self._is_ftp(packet):
            return "FTP"
        elif self._is_smtp(packet):
            return "SMTP"
        elif self._is_pop3(packet):
            return "POP3"
        elif self._is_imap(packet):
            return "IMAP"
        elif self._is_ssh(packet):
            return "SSH"
        elif self._is_telnet(packet):
            return "Telnet"
        elif self._is_dns(packet):
            return "DNS"
        elif self._is_dhcp(packet):
            return "DHCP"
        elif self._is_snmp(packet):
            return "SNMP"
        # 检查网络层协议
        elif self._is_icmp(packet):
            return "ICMP"
        # 检查传输层协议
        elif self._is_tcp(packet):
            return "TCP"
        elif self._is_udp(packet):
            return "UDP"
        # 检查网络层协议
        elif self._is_ipv4(packet):
            return "IPv4"
        # 检查数据链路层协议
        elif self._is_arp(packet):
            return "ARP"
        elif self._is_ethernet(packet):
            return "Ethernet"
        else:
            return "Unknown"
    
    def _is_ethernet(self, packet):
        from scapy.layers.l2 import Ether
        return packet.haslayer(Ether)
    
    def _is_ipv4(self, packet):
        from scapy.layers.inet import IP
        return packet.haslayer(IP)
    
    def _is_ipv6(self, packet):
        from scapy.layers.inet6 import IPv6
        return packet.haslayer(IPv6)
    
    def _is_tcp(self, packet):
        from scapy.layers.inet import TCP
        return packet.haslayer(TCP)
    
    def _is_udp(self, packet):
        from scapy.layers.inet import UDP
        return packet.haslayer(UDP)
    
    def _is_http(self, packet):
        """识别HTTP协议（端口80）"""
        from scapy.layers.inet import TCP
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport == 80 or tcp.sport == 80
        return False
    
    def _is_https(self, packet):
        """识别HTTPS协议（端口443）"""
        from scapy.layers.inet import TCP
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport == 443 or tcp.sport == 443
        return False
    
    def _is_ftp(self, packet):
        """识别FTP协议（端口21控制连接，端口20数据连接）"""
        from scapy.layers.inet import TCP
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport in [20, 21] or tcp.sport in [20, 21]
        return False
    
    def _is_smtp(self, packet):
        """识别SMTP协议（端口25）"""
        from scapy.layers.inet import TCP
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport == 25 or tcp.sport == 25
        return False
    
    def _is_pop3(self, packet):
        """识别POP3协议（端口110）"""
        from scapy.layers.inet import TCP
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport == 110 or tcp.sport == 110
        return False
    
    def _is_imap(self, packet):
        """识别IMAP协议（端口143）"""
        from scapy.layers.inet import TCP
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport == 143 or tcp.sport == 143
        return False
    
    def _is_ssh(self, packet):
        """识别SSH协议（端口22）"""
        from scapy.layers.inet import TCP
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport == 22 or tcp.sport == 22
        return False
    
    def _is_telnet(self, packet):
        """识别Telnet协议（端口23）"""
        from scapy.layers.inet import TCP
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport == 23 or tcp.sport == 23
        return False
    
    def _is_dns(self, packet):
        """识别DNS协议（端口53）"""
        from scapy.layers.inet import UDP, TCP
        if packet.haslayer(UDP):
            udp = packet.getlayer(UDP)
            return udp.dport == 53 or udp.sport == 53
        elif packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            return tcp.dport == 53 or tcp.sport == 53
        return False
    
    def _is_dhcp(self, packet):
        """识别DHCP协议（端口67/68）"""
        from scapy.layers.inet import UDP
        if packet.haslayer(UDP):
            udp = packet.getlayer(UDP)
            return (udp.dport == 67 and udp.sport == 68) or (udp.dport == 68 and udp.sport == 67)
        return False
    
    def _is_snmp(self, packet):
        """识别SNMP协议（端口161/162）"""
        from scapy.layers.inet import UDP
        if packet.haslayer(UDP):
            udp = packet.getlayer(UDP)
            return udp.dport in [161, 162] or udp.sport in [161, 162]
        return False
    
    def _is_icmp(self, packet):
        """识别ICMP协议"""
        from scapy.layers.inet import ICMP
        return packet.haslayer(ICMP)
    
    def _is_arp(self, packet):
        """识别ARP协议"""
        from scapy.layers.l2 import ARP
        return packet.haslayer(ARP)
