#!/usr/bin/env python3
# 协议解码模块

class ProtocolDecoder:
    def __init__(self):
        pass
    
    def decode_packet(self, packet, protocol):
        """
        解码数据包，根据协议类型提取关键信息
        :param packet: 数据包对象
        :param protocol: 协议名称
        :return: 解码后的信息字符串
        """
        info = ""
        
        # 首先解码基础的网络层和传输层信息
        info += self._decode_ip(packet)
        info += self._decode_transport(packet)
        
        # 然后解码应用层协议
        if protocol == "HTTP":
            info += self._decode_http(packet)
        elif protocol == "HTTPS":
            info += "HTTPS协议: 加密传输，无法直接查看内容\n"
        elif protocol == "FTP":
            info += self._decode_ftp(packet)
        elif protocol == "SMTP":
            info += self._decode_smtp(packet)
        elif protocol == "POP3":
            info += self._decode_pop3(packet)
        elif protocol == "IMAP":
            info += self._decode_imap(packet)
        elif protocol == "SSH":
            info += self._decode_ssh(packet)
        elif protocol == "Telnet":
            info += self._decode_telnet(packet)
        elif protocol == "DNS":
            info += self._decode_dns(packet)
        elif protocol == "DHCP":
            info += self._decode_dhcp(packet)
        elif protocol == "SNMP":
            info += self._decode_snmp(packet)
        elif protocol == "ICMP":
            info += self._decode_icmp(packet)
        elif protocol == "ARP":
            info += self._decode_arp(packet)
        
        return info
    
    def _decode_ip(self, packet):
        """解码IP层信息，支持IPv4和IPv6"""
        info = ""
        from scapy.layers.inet import IP
        from scapy.layers.inet6 import IPv6
        
        if packet.haslayer(IP):
            ip = packet.getlayer(IP)
            info += f"源IP: {ip.src}\n"
            info += f"目的IP: {ip.dst}\n"
            info += f"IP协议: IPv4\n"
            info += f"TTL: {ip.ttl}\n"
            info += f"总长度: {ip.len}\n"
        elif packet.haslayer(IPv6):
            ipv6 = packet.getlayer(IPv6)
            info += f"源IP: {ipv6.src}\n"
            info += f"目的IP: {ipv6.dst}\n"
            info += f"IP协议: IPv6\n"
            info += f"跳数限制: {ipv6.hlim}\n"
            info += f"有效负载长度: {ipv6.plen}\n"
            info += f"下一跳头部: {ipv6.nh}\n"
        return info
    
    def _decode_transport(self, packet):
        """解码传输层信息"""
        info = ""
        from scapy.layers.inet import TCP, UDP
        
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            info += f"传输层协议: TCP\n"
            info += f"源端口: {tcp.sport}\n"
            info += f"目的端口: {tcp.dport}\n"
            info += f"序列号: {tcp.seq}\n"
            info += f"确认号: {tcp.ack}\n"
            info += f"窗口大小: {tcp.window}\n"
            
            # 解析TCP标志位
            flags = []
            if tcp.flags.F:
                flags.append("FIN")
            if tcp.flags.S:
                flags.append("SYN")
            if tcp.flags.R:
                flags.append("RST")
            if tcp.flags.P:
                flags.append("PSH")
            if tcp.flags.A:
                flags.append("ACK")
            if tcp.flags.U:
                flags.append("URG")
            info += f"标志位: {','.join(flags)}\n"
        elif packet.haslayer(UDP):
            udp = packet.getlayer(UDP)
            info += f"传输层协议: UDP\n"
            info += f"源端口: {udp.sport}\n"
            info += f"目的端口: {udp.dport}\n"
            info += f"长度: {udp.len}\n"
        return info
    
    def _decode_http(self, packet):
        """解码HTTP协议信息，包括完整的请求/响应体"""
        info = "HTTP协议:\n"
        from scapy.layers.inet import TCP
        
        # 获取TCP负载
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            payload = bytes(tcp.payload)
            
            if payload:
                try:
                    http_data = payload.decode('utf-8', errors='ignore')
                    lines = http_data.split('\n')
                    
                    # 解析请求或响应行
                    if lines:
                        first_line = lines[0].strip()
                        if first_line.startswith(('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH')):
                            # HTTP请求
                            info += f"  请求方法: {first_line.split()[0]}\n"
                            info += f"  请求URL: {first_line.split()[1]}\n"
                            info += f"  HTTP版本: {first_line.split()[2]}\n"
                        elif first_line.startswith('HTTP/'):
                            # HTTP响应
                            info += f"  响应行: {first_line}\n"
                            if len(first_line.split()) >= 2:
                                info += f"  状态码: {first_line.split()[1]}\n"
                                if len(first_line.split()) >= 3:
                                    info += f"  状态描述: {' '.join(first_line.split()[2:])}\n"
                    
                    # 解析请求头/响应头
                    headers = {}
                    body_start = 1
                    info += "  头部信息:\n"
                    for i, line in enumerate(lines[1:]):
                        line = line.strip()
                        if line == '':
                            # 空行表示头部结束
                            body_start = i + 2
                            break
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            headers[key] = value
                            info += f"    {key}: {value}\n"
                            # 提取关键头部信息
                            if key.lower() == 'user-agent':
                                info += f"    【User-Agent: {value}】\n"
                            elif key.lower() == 'content-type':
                                info += f"    【Content-Type: {value}】\n"
                            elif key.lower() == 'content-length':
                                info += f"    【Content-Length: {value}】\n"
                    
                    # 解析请求体/响应体
                    body = '\n'.join(lines[body_start:]).strip()
                    if body:
                        info += "  请求体/响应体:\n"
                        # 检查内容类型
                        content_type = headers.get('Content-Type', '')
                        if 'application/json' in content_type:
                            # 尝试格式化JSON
                            try:
                                import json
                                json_data = json.loads(body)
                                formatted_body = json.dumps(json_data, indent=2)
                                info += f"    {formatted_body[:500]}\n"  # 只显示前500个字符
                                if len(formatted_body) > 500:
                                    info += f"    ... (内容过长，已截断)\n"
                            except:
                                info += f"    {body[:500]}\n"  # 只显示前500个字符
                                if len(body) > 500:
                                    info += f"    ... (内容过长，已截断)\n"
                        elif 'text/' in content_type:
                            # 文本内容，直接显示
                            info += f"    {body[:500]}\n"  # 只显示前500个字符
                            if len(body) > 500:
                                info += f"    ... (内容过长，已截断)\n"
                        else:
                            # 二进制内容，显示长度
                            info += f"    【二进制数据，长度: {len(body)} 字节】\n"
                    
                except Exception as e:
                    info += f"  解析HTTP数据失败: {e}\n"
        
        return info
    
    def _decode_ftp(self, packet):
        """解码FTP协议信息"""
        info = "FTP协议:\n"
        from scapy.layers.inet import TCP
        
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            payload = bytes(tcp.payload)
            if payload:
                try:
                    ftp_data = payload.decode('utf-8', errors='ignore').strip()
                    info += f"  FTP数据: {ftp_data}\n"
                except:
                    pass
        return info
    
    def _decode_smtp(self, packet):
        """解码SMTP协议信息"""
        info = "SMTP协议:\n"
        from scapy.layers.inet import TCP
        
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            payload = bytes(tcp.payload)
            if payload:
                try:
                    smtp_data = payload.decode('utf-8', errors='ignore').strip()
                    info += f"  SMTP数据: {smtp_data}\n"
                except:
                    pass
        return info
    
    def _decode_pop3(self, packet):
        """解码POP3协议信息"""
        info = "POP3协议:\n"
        from scapy.layers.inet import TCP
        
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            payload = bytes(tcp.payload)
            if payload:
                try:
                    pop3_data = payload.decode('utf-8', errors='ignore').strip()
                    info += f"  POP3数据: {pop3_data}\n"
                except:
                    pass
        return info
    
    def _decode_imap(self, packet):
        """解码IMAP协议信息"""
        info = "IMAP协议:\n"
        from scapy.layers.inet import TCP
        
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            payload = bytes(tcp.payload)
            if payload:
                try:
                    imap_data = payload.decode('utf-8', errors='ignore').strip()
                    info += f"  IMAP数据: {imap_data}\n"
                except:
                    pass
        return info
    
    def _decode_ssh(self, packet):
        """解码SSH协议信息"""
        info = "SSH协议:\n"
        from scapy.layers.inet import TCP
        
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            payload = bytes(tcp.payload)
            if payload:
                # SSH握手阶段，提取协议版本信息
                try:
                    ssh_data = payload.decode('utf-8', errors='ignore').strip()
                    if ssh_data.startswith('SSH-'):
                        info += f"  SSH版本: {ssh_data}\n"
                    else:
                        info += f"  SSH数据: 加密传输，无法直接查看\n"
                except:
                    info += f"  SSH数据: 加密传输，无法直接查看\n"
        return info
    
    def _decode_telnet(self, packet):
        """解码Telnet协议信息"""
        info = "Telnet协议:\n"
        from scapy.layers.inet import TCP
        
        if packet.haslayer(TCP):
            tcp = packet.getlayer(TCP)
            payload = bytes(tcp.payload)
            if payload:
                try:
                    # 过滤Telnet控制字符
                    telnet_data = ''
                    i = 0
                    while i < len(payload):
                        if payload[i] == 255:  # IAC
                            i += 3  # 跳过控制序列
                        else:
                            telnet_data += chr(payload[i])
                            i += 1
                    
                    if telnet_data:
                        info += f"  Telnet数据: {telnet_data.strip()}\n"
                    else:
                        info += f"  Telnet控制数据\n"
                except:
                    pass
        return info
    
    def _decode_dns(self, packet):
        """解码DNS协议信息"""
        info = "DNS协议:\n"
        from scapy.layers.dns import DNS
        
        if packet.haslayer(DNS):
            dns = packet.getlayer(DNS)
            info += f"  DNS ID: {dns.id}\n"
            info += f"  标志: {'请求' if dns.qr == 0 else '响应'}\n"
            info += f"  操作码: {dns.opcode}\n"
            info += f"  响应码: {dns.rcode}\n"
            
            # 解析DNS查询
            if dns.qr == 0:
                info += f"  查询数量: {dns.qdcount}\n"
                for i in range(dns.qdcount):
                    qd = dns.qd[i]
                    info += f"  查询: {qd.qname.decode('utf-8')} 类型: {qd.qtype} 类: {qd.qclass}\n"
            # 解析DNS响应
            else:
                info += f"  查询数量: {dns.qdcount}\n"
                info += f"  回答数量: {dns.ancount}\n"
                info += f"  授权记录数量: {dns.nscount}\n"
                info += f"  附加记录数量: {dns.arcount}\n"
                
                # 查询部分
                for i in range(dns.qdcount):
                    qd = dns.qd[i]
                    info += f"  查询: {qd.qname.decode('utf-8')} 类型: {qd.qtype} 类: {qd.qclass}\n"
                
                # 回答部分
                for i in range(dns.ancount):
                    an = dns.an[i]
                    info += f"  回答: {an.rrname.decode('utf-8')} 类型: {an.type} 类: {an.rclass} TTL: {an.ttl}\n"
                    if an.type == 1:  # A记录
                        info += f"    IP: {an.rdata}\n"
                    elif an.type == 28:  # AAAA记录
                        info += f"    IPv6: {an.rdata}\n"
                    elif an.type == 5:  # CNAME记录
                        info += f"    CNAME: {an.rdata.decode('utf-8')}\n"
        return info
    
    def _decode_dhcp(self, packet):
        """解码DHCP协议信息"""
        info = "DHCP协议:\n"
        from scapy.layers.dhcp import DHCP
        from scapy.layers.inet import UDP
        
        if packet.haslayer(DHCP) and packet.haslayer(UDP):
            dhcp = packet.getlayer(DHCP)
            udp = packet.getlayer(UDP)
            
            # 判断是DHCP请求还是响应
            if udp.sport == 68 and udp.dport == 67:
                info += f"  DHCP类型: 请求\n"
            else:
                info += f"  DHCP类型: 响应\n"
            
            # 解析DHCP选项
            for opt in dhcp.options:
                if isinstance(opt, tuple):
                    opt_type = opt[0]
                    if opt_type == 'message-type':
                        msg_type = opt[1]
                        msg_types = {
                            1: 'DHCPDISCOVER',
                            2: 'DHCPOFFER',
                            3: 'DHCPREQUEST',
                            4: 'DHCPDECLINE',
                            5: 'DHCPACK',
                            6: 'DHCPNAK',
                            7: 'DHCPRELEASE',
                            8: 'DHCPINFORM'
                        }
                        info += f"  消息类型: {msg_types.get(msg_type, f'未知 ({msg_type})')}\n"
                    elif opt_type == 'server_id':
                        info += f"  DHCP服务器: {opt[1]}\n"
                    elif opt_type == 'requested_addr':
                        info += f"  请求IP: {opt[1]}\n"
                    elif opt_type == 'hostname':
                        info += f"  主机名: {opt[1].decode('utf-8')}\n"
                    elif opt_type == 'subnet_mask':
                        info += f"  子网掩码: {opt[1]}\n"
                    elif opt_type == 'router':
                        info += f"  网关: {opt[1]}\n"
                    elif opt_type == 'name_server':
                        info += f"  DNS服务器: {opt[1]}\n"
        return info
    
    def _decode_snmp(self, packet):
        """解码SNMP协议信息"""
        info = "SNMP协议:\n"
        from scapy.layers.inet import UDP
        
        if packet.haslayer(UDP):
            udp = packet.getlayer(UDP)
            info += f"  SNMP端口: {'请求' if udp.sport == 161 else '陷阱'} - {'服务器' if udp.dport == 161 else '管理器'}\n"
            info += f"  SNMP数据: 需使用SNMP解析器深入分析\n"
        return info
    
    def _decode_icmp(self, packet):
        """解码ICMP协议信息"""
        info = "ICMP协议:\n"
        from scapy.layers.inet import ICMP
        
        if packet.haslayer(ICMP):
            icmp = packet.getlayer(ICMP)
            info += f"  ICMP类型: {icmp.type}\n"
            info += f"  ICMP代码: {icmp.code}\n"
            
            # ICMP类型描述
            icmp_types = {
                0: '回显应答',
                3: '目标不可达',
                4: '源抑制',
                5: '重定向',
                8: '回显请求',
                11: '超时',
                12: '参数问题',
                13: '时间戳请求',
                14: '时间戳应答',
                17: '地址掩码请求',
                18: '地址掩码应答'
            }
            info += f"  类型描述: {icmp_types.get(icmp.type, f'未知 ({icmp.type})')}\n"
            
            if icmp.type == 8 or icmp.type == 0:  # 回显请求/应答
                info += f"  标识符: {icmp.id}\n"
                info += f"  序列号: {icmp.seq}\n"
        return info
    
    def _decode_arp(self, packet):
        """解码ARP协议信息"""
        info = "ARP协议:\n"
        from scapy.layers.l2 import ARP
        
        if packet.haslayer(ARP):
            arp = packet.getlayer(ARP)
            info += f"  ARP操作: {'请求' if arp.op == 1 else '响应'}\n"
            info += f"  源MAC: {arp.hwsrc}\n"
            info += f"  源IP: {arp.psrc}\n"
            info += f"  目的MAC: {arp.hwdst}\n"
            info += f"  目的IP: {arp.pdst}\n"
        return info
