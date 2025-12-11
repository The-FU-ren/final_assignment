#!/usr/bin/env python3
# 协议分析软件图形界面

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from packet_capture import PacketCapture
from protocol_identifier import ProtocolIdentifier
from protocol_decoder import ProtocolDecoder
from packet_storage import PacketStorage
import threading

class ProtocolAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("协议分析软件")
        self.root.geometry("1000x600")
        
        # 初始化各模块
        self.capture = PacketCapture()
        self.identifier = ProtocolIdentifier()
        self.decoder = ProtocolDecoder()
        self.storage = PacketStorage()
        
        # 捕获状态
        self.is_capturing = False
        self.captured_packets = []
        self.original_packets = []  # 保存原始数据包，用于筛选功能
        self.packet_count = 0
        
        # 创建菜单
        self.create_menu()
        
        # 创建主框架
        self.create_main_frame()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 更新可用接口列表
        self.update_interfaces()
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开PCAP文件", command=self.open_pcap_file)
        file_menu.add_command(label="保存PCAP文件", command=self.save_pcap_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 捕获菜单
        capture_menu = tk.Menu(menubar, tearoff=0)
        capture_menu.add_command(label="开始捕获", command=self.start_capture)
        capture_menu.add_command(label="停止捕获", command=self.stop_capture)
        menubar.add_cascade(label="捕获", menu=capture_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_frame(self):
        """创建主框架"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. 捕获控制区
        control_frame = ttk.LabelFrame(main_frame, text="捕获控制", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 接口选择
        ttk.Label(control_frame, text="网络接口:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.interface_var = tk.StringVar()
        self.interface_combo = ttk.Combobox(control_frame, textvariable=self.interface_var, width=60)
        self.interface_combo.grid(row=0, column=1, columnspan=3, padx=5, pady=5)
        
        # 刷新接口按钮
        ttk.Button(control_frame, text="刷新", command=self.update_interfaces).grid(row=0, column=4, padx=5, pady=5)
        
        # 过滤规则
        ttk.Label(control_frame, text="过滤规则:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.filter_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.filter_var, width=70).grid(row=1, column=1, columnspan=4, padx=5, pady=5)
        
        # 捕获按钮
        ttk.Button(control_frame, text="开始捕获", command=self.start_capture).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="停止捕获", command=self.stop_capture).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="清空列表", command=lambda: self.clear_packets(clear_original=True)).grid(row=2, column=2, padx=5, pady=5)
        
        # 筛选功能
        ttk.Label(control_frame, text="筛选条件:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.filter_text_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.filter_text_var, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="应用筛选", command=self.apply_filter).grid(row=3, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="清除筛选", command=self.clear_filter).grid(row=3, column=3, padx=5, pady=5)
        
        # 2. 数据包列表和详情区
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # 数据包列表
        list_frame = ttk.LabelFrame(bottom_frame, text="数据包列表", padding="10")
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 创建树形视图
        columns = ("no", "time", "source", "destination", "protocol", "length", "info")
        self.packet_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # 设置列宽和标题
        self.packet_tree.column("no", width=50, anchor=tk.CENTER)
        self.packet_tree.column("time", width=120, anchor=tk.CENTER)
        self.packet_tree.column("source", width=120, anchor=tk.W)
        self.packet_tree.column("destination", width=120, anchor=tk.W)
        self.packet_tree.column("protocol", width=80, anchor=tk.CENTER)
        self.packet_tree.column("length", width=80, anchor=tk.E)
        self.packet_tree.column("info", width=200, anchor=tk.W)
        
        # 设置列标题
        self.packet_tree.heading("no", text="编号")
        self.packet_tree.heading("time", text="时间")
        self.packet_tree.heading("source", text="源地址")
        self.packet_tree.heading("destination", text="目的地址")
        self.packet_tree.heading("protocol", text="协议")
        self.packet_tree.heading("length", text="长度")
        self.packet_tree.heading("info", text="信息")
        
        # 绑定选中事件
        self.packet_tree.bind("<<TreeviewSelect>>", self.on_packet_select)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.packet_tree.yview)
        self.packet_tree.configure(yscroll=scrollbar.set)
        self.packet_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 数据包详情
        detail_frame = ttk.LabelFrame(bottom_frame, text="数据包详情", padding="10")
        detail_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建文本框显示详情
        self.detail_text = tk.Text(detail_frame, wrap=tk.WORD, width=50, height=20)
        scrollbar_y = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail_text.yview)
        scrollbar_x = ttk.Scrollbar(detail_frame, orient=tk.HORIZONTAL, command=self.detail_text.xview)
        self.detail_text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.detail_text.pack(fill=tk.BOTH, expand=True)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_interfaces(self):
        """更新可用网络接口列表，显示更友好的名称"""
        from scapy.all import IFACES, get_if_list
        
        # 获取所有网络接口的详细信息
        self.interfaces = {}
        friendly_names = []
        if_list = get_if_list()
        
        for name, iface in IFACES.items():
            # 构建友好的接口名称
            friendly_name = f"{iface.description} ({name})"
            if iface.ip:
                friendly_name += f" - {iface.ip}"
            
            friendly_names.append(friendly_name)
            # 保存实际的接口名称，用于捕获
            self.interfaces[friendly_name] = name
        
        self.interface_combo['values'] = friendly_names
        if friendly_names:
            self.interface_combo.current(0)
        self.status_var.set(f"已加载 {len(friendly_names)} 个网络接口")
    
    def start_capture(self):
        """开始捕获数据包"""
        if self.is_capturing:
            messagebox.showwarning("警告", "捕获已在进行中")
            return
        
        selected_friendly_name = self.interface_var.get()
        if not selected_friendly_name:
            messagebox.showerror("错误", "请选择网络接口")
            return
        
        # 获取实际的接口名称
        interface = self.interfaces.get(selected_friendly_name, selected_friendly_name)
        
        filter_rule = self.filter_var.get()
        
        # 开始捕获线程
        self.is_capturing = True
        self.status_var.set(f"正在捕获数据包...")
        
        # 清空之前的数据包
        self.clear_packets(clear_original=True)
        
        # 创建捕获线程
        capture_thread = threading.Thread(
            target=self.capture_packets_thread,
            args=(interface, filter_rule)
        )
        capture_thread.daemon = True
        capture_thread.start()
    
    def capture_packets_thread(self, interface, filter_rule):
        """捕获数据包的线程函数"""
        try:
            # 使用实时捕获功能
            self.capture.start_live_capture(
                interface=interface,
                callback=self.on_packet_captured,
                filter_rule=filter_rule
            )
        except Exception as e:
            messagebox.showerror("捕获错误", f"捕获失败: {str(e)}")
            self.status_var.set("捕获失败")
            self.is_capturing = False
    
    def on_packet_captured(self, packet):
        """捕获到单个数据包时的回调函数"""
        # 将数据包添加到列表
        self.captured_packets.append(packet)
        self.original_packets.append(packet)  # 保存到原始数据包列表
        self.packet_count += 1
        
        # 获取当前数据包在列表中的索引作为唯一标识符
        packet_index = len(self.captured_packets) - 1
        
        # 识别协议
        protocol = self.identifier.identify_protocol(packet)
        
        # 获取基本信息
        source = ""
        destination = ""
        length = len(packet)
        info = ""
        
        # 获取IP层信息
        from scapy.layers.inet import IP
        from scapy.layers.inet import TCP, UDP
        
        if packet.haslayer(IP):
            ip = packet.getlayer(IP)
            source = ip.src
            destination = ip.dst
        
        # 获取传输层信息
        if protocol == "HTTP":
            if packet.haslayer(TCP):
                tcp = packet.getlayer(TCP)
                payload = bytes(tcp.payload)
                if payload:
                    try:
                        http_data = payload.decode('utf-8', errors='ignore')
                        first_line = http_data.split('\n')[0].strip()
                        info = first_line[:50]  # 只显示前50个字符
                    except:
                        info = "HTTP数据"
        elif protocol == "DNS":
            if packet.haslayer(TCP) or packet.haslayer(UDP):
                info = "DNS请求/响应"
        elif protocol == "DHCP":
            info = "DHCP请求/响应"
        elif protocol == "ICMP":
            info = "ICMP消息"
        elif protocol == "ARP":
            info = "ARP请求/响应"
        
        # 在GUI线程中更新数据包列表
        def update_gui():
            try:
                # 插入到树形视图，使用列表索引作为唯一iid
                self.packet_tree.insert("", tk.END, iid=packet_index, values=(
                    self.packet_count, "0.000000", source, destination, protocol, length, info
                ))
                # 更新状态栏
                self.status_var.set(f"正在捕获数据包... 已捕获 {self.packet_count} 个数据包")
            except Exception as e:
                print(f"更新GUI失败: {e}")
        
        self.root.after(0, update_gui)
    
    def stop_capture(self):
        """停止捕获数据包"""
        if self.is_capturing:
            self.capture.stop_live_capture()
            self.is_capturing = False
            self.status_var.set(f"捕获已停止，共捕获 {self.packet_count} 个数据包")
        else:
            messagebox.showinfo("提示", "当前未在捕获数据包")
    
    def display_captured_packets(self):
        """显示捕获的数据包"""
        for i, packet in enumerate(self.captured_packets):
            protocol = self.identifier.identify_protocol(packet)
            
            # 获取基本信息
            source = ""
            destination = ""
            length = len(packet)
            info = ""
            
            # 获取IP层信息
            from scapy.layers.inet import IP
            from scapy.layers.inet import TCP, UDP
            
            if packet.haslayer(IP):
                ip = packet.getlayer(IP)
                source = ip.src
                destination = ip.dst
            
            # 获取传输层信息
            if protocol == "HTTP":
                if packet.haslayer(TCP):
                    payload = bytes(packet.getlayer(TCP).payload)
                    if payload:
                        try:
                            http_data = payload.decode('utf-8', errors='ignore')
                            first_line = http_data.split('\n')[0].strip()
                            info = first_line[:50]  # 只显示前50个字符
                        except:
                            info = "HTTP数据"
            
            # 插入到树形视图
            self.packet_tree.insert("", tk.END, iid=i, values=(
                i+1, "0.000000", source, destination, protocol, length, info
            ))
    
    def on_packet_select(self, event):
        """处理数据包选中事件"""
        selected_items = self.packet_tree.selection()
        if not selected_items:
            return
        
        item = selected_items[0]
        packet_index = int(item)
        
        if 0 <= packet_index < len(self.captured_packets):
            packet = self.captured_packets[packet_index]
            protocol = self.identifier.identify_protocol(packet)
            decoded_info = self.decoder.decode_packet(packet, protocol)
            
            # 显示详情
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(tk.END, f"=== 数据包 {packet_index+1} 详情 ===\n\n")
            self.detail_text.insert(tk.END, decoded_info)
    
    def open_pcap_file(self):
        """打开PCAP文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("PCAP文件", "*.pcap"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                self.status_var.set(f"正在读取PCAP文件...")
                packets = self.storage.read_pcap(file_path)
                # 先清空，再设置数据包列表
                self.clear_packets(clear_original=True)
                self.captured_packets = packets
                self.original_packets = packets.copy()  # 保存原始数据包
                self.display_captured_packets()
                self.status_var.set(f"已读取 {len(packets)} 个数据包")
            except Exception as e:
                messagebox.showerror("读取错误", f"读取PCAP文件失败: {str(e)}")
                self.status_var.set("读取失败")
    
    def save_pcap_file(self):
        """保存PCAP文件"""
        if not self.captured_packets:
            messagebox.showwarning("警告", "没有可保存的数据包")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pcap",
            filetypes=[("PCAP文件", "*.pcap"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                self.status_var.set(f"正在保存PCAP文件...")
                if self.storage.save_pcap(file_path, self.captured_packets):
                    self.status_var.set(f"已保存 {len(self.captured_packets)} 个数据包")
                    messagebox.showinfo("成功", "PCAP文件保存成功")
                else:
                    raise Exception("保存失败")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存PCAP文件失败: {str(e)}")
                self.status_var.set("保存失败")
    
    def clear_packets(self, clear_original=False):
        """清空数据包列表
        
        Args:
            clear_original: 是否清空原始数据包，默认为False
        """
        for item in self.packet_tree.get_children():
            self.packet_tree.delete(item)
        self.detail_text.delete(1.0, tk.END)
        self.captured_packets = []
        if clear_original:
            self.original_packets = []  # 仅在需要时清空原始数据包
        self.packet_count = 0
        self.status_var.set("数据包列表已清空")
    
    def show_about(self):
        """显示关于信息"""
        about_text = "协议分析软件 v1.0\n\n" \
                     "一款功能强大的网络协议分析工具，支持多种常见网络协议的识别、解码和存储。\n\n" \
                     "功能特性：\n" \
                     "- 协议识别：TCP/IP、UDP、HTTP、HTTPS、FTP、SMTP、POP3、IMAP等\n" \
                     "- 数据捕获：支持从网络接口捕获数据包\n" \
                     "- 过滤规则：支持BPF过滤语法\n" \
                     "- 协议解码：深入解析各层协议信息\n" \
                     "- 数据存储：支持PCAP格式文件的读写\n" \
                     "- 数据包筛选：支持按协议、IP地址、端口号等筛选\n\n" \
                     "开发语言：Python 3\n" \
                     "核心库：Scapy 2.5.0"
        
        messagebox.showinfo("关于", about_text)
    
    def apply_filter(self):
        """应用筛选条件"""
        filter_text = self.filter_text_var.get().strip().lower()
        if not filter_text:
            messagebox.showinfo("提示", "请输入筛选条件")
            return
        
        try:
            # 过滤数据包
            filtered_packets = []
            for packet in self.original_packets:
                # 识别协议
                protocol = self.identifier.identify_protocol(packet).lower()
                
                # 获取基本信息
                source = ""
                destination = ""
                src_port = ""
                dst_port = ""
                
                # 获取IP层信息
                from scapy.layers.inet import IP
                from scapy.layers.inet import TCP, UDP
                
                if packet.haslayer(IP):
                    ip = packet.getlayer(IP)
                    source = ip.src
                    destination = ip.dst
                
                # 获取端口信息
                if packet.haslayer(TCP):
                    tcp = packet.getlayer(TCP)
                    src_port = str(tcp.sport)
                    dst_port = str(tcp.dport)
                elif packet.haslayer(UDP):
                    udp = packet.getlayer(UDP)
                    src_port = str(udp.sport)
                    dst_port = str(udp.dport)
                
                # 检查筛选条件
                if (
                    filter_text in protocol or
                    filter_text in source.lower() or
                    filter_text in destination.lower() or
                    filter_text in src_port or
                    filter_text in dst_port
                ):
                    filtered_packets.append(packet)
            
            # 更新数据包列表
            self.clear_packets()  # 保留原始数据包，只清空当前显示的数据包
            self.captured_packets = filtered_packets
            self.display_captured_packets()
            self.status_var.set(f"已筛选出 {len(filtered_packets)} 个数据包")
        except Exception as e:
            messagebox.showerror("筛选错误", f"筛选失败: {str(e)}")
            self.status_var.set("筛选失败")
    
    def clear_filter(self):
        """清除筛选条件"""
        # 清空筛选条件文本
        self.filter_text_var.set("")
        # 恢复显示所有原始数据包
        self.clear_packets()  # 保留原始数据包，只清空当前显示的数据包
        self.captured_packets = self.original_packets.copy()
        self.display_captured_packets()
        self.status_var.set(f"已显示所有 {len(self.original_packets)} 个数据包")
    
    def run(self):
        """运行GUI程序"""
        self.root.mainloop()

# 运行GUI程序
if __name__ == "__main__":
    app = ProtocolAnalyzerGUI()
    app.run()
