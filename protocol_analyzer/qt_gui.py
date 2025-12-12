#!/usr/bin/env python3
# 协议分析软件图形界面 - PyQt5版本

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QTreeView, QTableWidget, QTableWidgetItem, QTextEdit,
    QPushButton, QLineEdit, QLabel, QComboBox, QFileDialog, QStatusBar,
    QMenuBar, QMenu, QAction, QToolBar, QGroupBox, QGridLayout, 
    QSplitter, QFrame, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSortFilterProxyModel, QModelIndex
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont, QIcon, QPalette, QColor
import sys
import os

# 导入原有功能模块
from packet_capture import PacketCapture
from protocol_identifier import ProtocolIdentifier
from protocol_decoder import ProtocolDecoder
from packet_storage import PacketStorage

class CaptureThread(QThread):
    """捕获数据包的线程"""
    packet_captured = pyqtSignal(object)
    capture_stopped = pyqtSignal()
    capture_error = pyqtSignal(str)
    
    def __init__(self, capture, interface, filter_rule):
        super().__init__()
        self.capture = capture
        self.interface = interface
        self.filter_rule = filter_rule
        self.is_running = True
    
    def run(self):
        try:
            def callback(packet):
                if self.is_running:
                    self.packet_captured.emit(packet)
            
            self.capture.start_live_capture(
                self.interface, 
                callback, 
                self.filter_rule
            )
            self.capture_stopped.emit()
        except Exception as e:
            self.capture_error.emit(str(e))
    
    def stop(self):
        self.is_running = False
        self.capture.stop_live_capture()
        self.wait()

class ProtocolAnalyzerGUI(QMainWindow):
    """协议分析软件主界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("协议分析软件")
        
        # 设置窗口大小为屏幕的70%
        screen_geometry = QApplication.desktop().screenGeometry()
        self.resize(int(screen_geometry.width() * 0.7), int(screen_geometry.height() * 0.7))
        
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
        
        # 主题设置
        self.current_theme = "light"
        self.themes = {
            "light": {
                "background": QColor(255, 255, 255),
                "text": QColor(0, 0, 0),
                "frame": QColor(240, 240, 240),
                "button": QColor(220, 220, 220),
                "button_text": QColor(0, 0, 0),
                "text_edit": QColor(255, 255, 255),
                "text_edit_text": QColor(0, 0, 0),
                "table_header": QColor(220, 220, 220),
                "table_alternate": QColor(240, 240, 240),
            },
            "dark": {
                "background": QColor(45, 45, 45),
                "text": QColor(255, 255, 255),
                "frame": QColor(61, 61, 61),
                "button": QColor(77, 77, 77),
                "button_text": QColor(255, 255, 255),
                "text_edit": QColor(45, 45, 45),
                "text_edit_text": QColor(255, 255, 255),
                "table_header": QColor(77, 77, 77),
                "table_alternate": QColor(55, 55, 55),
            }
        }
        
        # 初始化UI
        self.init_ui()
        self.init_theme()
        self.update_interfaces()
        
        # 启用拖拽支持
        self.setAcceptDrops(True)
    
    def init_ui(self):
        """初始化UI界面，优化响应式布局"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 工具栏
        self.create_toolbar()
        
        # 控制面板
        self.create_control_panel()
        main_layout.addWidget(self.control_group)
        
        # 主分割器（垂直方向）
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setHandleWidth(10)
        main_splitter.setOpaqueResize(True)
        
        # 标签页分割器（水平方向）
        tab_splitter = QSplitter(Qt.Horizontal)
        tab_splitter.setHandleWidth(10)
        
        # 标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumSize(400, 300)
        
        # 1. 数据包列表标签
        self.create_packet_list_tab()
        
        # 2. 统计视图标签
        self.create_statistics_tab()
        
        # 3. 接口状态标签
        self.create_interface_status_tab()
        
        tab_splitter.addWidget(self.tab_widget)
        
        # 4. 数据包详情
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        self.create_packet_detail()
        detail_layout.addWidget(self.detail_group)
        detail_widget.setMinimumSize(300, 300)
        
        tab_splitter.addWidget(detail_widget)
        
        # 设置分割器初始比例
        tab_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        main_splitter.addWidget(tab_splitter)
        
        # 将主分割器添加到主布局
        main_layout.addWidget(main_splitter)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 菜单
        self.create_menu()
        
        # 绑定快捷键
        self.bind_shortcuts()
    
    def create_menu(self):
        """创建菜单栏"""
        menu_bar = self.menuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu("&文件")
        
        open_action = QAction("&打开PCAP文件", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_pcap_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("&保存PCAP文件", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_pcap_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("&退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 捕获菜单
        capture_menu = menu_bar.addMenu("&捕获")
        
        start_capture_action = QAction("&开始捕获", self)
        start_capture_action.setShortcut("Ctrl+G")
        start_capture_action.triggered.connect(self.start_capture)
        capture_menu.addAction(start_capture_action)
        
        stop_capture_action = QAction("&停止捕获", self)
        stop_capture_action.setShortcut("Ctrl+H")
        stop_capture_action.triggered.connect(self.stop_capture)
        capture_menu.addAction(stop_capture_action)
        
        clear_action = QAction("&清空列表", self)
        clear_action.triggered.connect(self.clear_packets)
        capture_menu.addAction(clear_action)
        
        # 视图菜单
        view_menu = menu_bar.addMenu("&视图")
        
        theme_menu = view_menu.addMenu("&主题")
        
        light_theme_action = QAction("&浅色主题", self, checkable=True, checked=True)
        light_theme_action.triggered.connect(lambda: self.toggle_theme("light"))
        theme_menu.addAction(light_theme_action)
        
        dark_theme_action = QAction("&深色主题", self, checkable=True)
        dark_theme_action.triggered.connect(lambda: self.toggle_theme("dark"))
        theme_menu.addAction(dark_theme_action)
        
        self.theme_actions = {"light": light_theme_action, "dark": dark_theme_action}
        
        # 帮助菜单
        help_menu = menu_bar.addMenu("&帮助")
        
        about_action = QAction("&关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """创建工具栏，将核心的开始/停止捕获功能移到最前面"""
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)
        
        # 开始捕获按钮（核心功能移到最前面）
        self.start_capture_btn = QPushButton("开始捕获")
        self.start_capture_btn.clicked.connect(self.start_capture)
        toolbar.addWidget(self.start_capture_btn)
        
        # 停止捕获按钮
        self.stop_capture_btn = QPushButton("停止捕获")
        self.stop_capture_btn.clicked.connect(self.stop_capture)
        self.stop_capture_btn.setEnabled(False)
        toolbar.addWidget(self.stop_capture_btn)
        
        # 清空列表按钮
        clear_btn = QPushButton("清空列表")
        clear_btn.clicked.connect(self.clear_packets)
        toolbar.addWidget(clear_btn)
        
        toolbar.addSeparator()
        
        # 打开文件按钮
        open_btn = QPushButton("打开文件")
        open_btn.clicked.connect(self.open_pcap_file)
        toolbar.addWidget(open_btn)
        
        # 保存文件按钮
        save_btn = QPushButton("保存文件")
        save_btn.clicked.connect(self.save_pcap_file)
        toolbar.addWidget(save_btn)
        
        toolbar.addSeparator()
        
        # 主题切换按钮
        self.theme_btn = QPushButton("切换主题")
        self.theme_btn.clicked.connect(self.toggle_theme)
        toolbar.addWidget(self.theme_btn)
    
    def create_control_panel(self):
        """创建控制面板"""
        self.control_group = QGroupBox("捕获控制")
        control_layout = QGridLayout(self.control_group)
        
        # 接口选择
        control_layout.addWidget(QLabel("网络接口:"), 0, 0)
        self.interface_combo = QComboBox()
        self.interface_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_layout.addWidget(self.interface_combo, 0, 1, 1, 2)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.update_interfaces)
        control_layout.addWidget(refresh_btn, 0, 3)
        
        # 过滤规则
        control_layout.addWidget(QLabel("过滤规则:"), 1, 0)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("输入过滤规则，如 tcp port 80，可对已捕获的包进行筛选")
        self.filter_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_layout.addWidget(self.filter_edit, 1, 1, 1, 2)
        
        # 添加过滤按钮
        self.apply_filter_btn = QPushButton("应用过滤")
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        control_layout.addWidget(self.apply_filter_btn, 1, 3)
        
        # 添加清除过滤按钮
        self.clear_filter_btn = QPushButton("清除过滤")
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        control_layout.addWidget(self.clear_filter_btn, 1, 4)
        
        # 搜索框
        control_layout.addWidget(QLabel("搜索:"), 2, 0)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索数据包内容")
        self.search_edit.textChanged.connect(self.search_packets)
        self.search_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_layout.addWidget(self.search_edit, 2, 1)
        
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.search_packets)
        control_layout.addWidget(search_btn, 2, 2)
        
        clear_search_btn = QPushButton("清除")
        clear_search_btn.clicked.connect(self.clear_search)
        control_layout.addWidget(clear_search_btn, 2, 3)
        
        
    
    def create_packet_list_tab(self):
        """创建数据包列表标签"""
        packet_list_widget = QWidget()
        packet_list_layout = QVBoxLayout(packet_list_widget)
        
        # 数据包表格
        self.packet_table = QTableWidget()
        self.packet_table.setColumnCount(7)
        self.packet_table.setHorizontalHeaderLabels([
            "编号", "时间", "源地址", "目的地址", "协议", "长度", "信息"
        ])
        self.packet_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.packet_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.packet_table.setSortingEnabled(True)
        self.packet_table.itemSelectionChanged.connect(self.on_packet_selected)
        
        # 设置列宽
        column_widths = [80, 150, 150, 150, 100, 80, 200]
        for i, width in enumerate(column_widths):
            self.packet_table.setColumnWidth(i, width)
        
        packet_list_layout.addWidget(self.packet_table)
        self.tab_widget.addTab(packet_list_widget, "数据包列表")
    
    def create_statistics_tab(self):
        """创建统计视图标签，只包含文本统计"""
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        # 统计信息文本
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        # 更新统计信息
        self.update_statistics()
        
        self.tab_widget.addTab(stats_widget, "统计视图")
    
    def create_interface_status_tab(self):
        """创建接口状态标签"""
        interface_widget = QWidget()
        interface_layout = QVBoxLayout(interface_widget)
        
        # 接口状态文本
        self.interface_text = QTextEdit()
        self.interface_text.setReadOnly(True)
        interface_layout.addWidget(self.interface_text)
        
        # 更新接口状态
        self.update_interface_status()
        
        self.tab_widget.addTab(interface_widget, "接口状态")
    
    def create_packet_detail(self):
        """创建数据包详情"""
        self.detail_group = QGroupBox("数据包详情")
        detail_layout = QVBoxLayout(self.detail_group)
        
        # 数据包详情文本
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setLineWrapMode(QTextEdit.NoWrap)
        detail_layout.addWidget(self.detail_text)
    
    def init_theme(self):
        """初始化主题"""
        self.apply_theme()
    
    def apply_theme(self):
        """应用主题"""
        theme = self.themes[self.current_theme]
        
        # 设置应用程序主题
        palette = QPalette()
        palette.setColor(QPalette.Window, theme["background"])
        palette.setColor(QPalette.WindowText, theme["text"])
        palette.setColor(QPalette.Base, theme["frame"])
        palette.setColor(QPalette.AlternateBase, theme["table_alternate"])
        palette.setColor(QPalette.ToolTipBase, theme["background"])
        palette.setColor(QPalette.ToolTipText, theme["text"])
        palette.setColor(QPalette.Text, theme["text"])
        palette.setColor(QPalette.Button, theme["button"])
        palette.setColor(QPalette.ButtonText, theme["button_text"])
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        QApplication.setPalette(palette)
        
        # 更新各部件的样式
        self.update_widget_styles()
    
    def update_widget_styles(self):
        """更新部件样式"""
        theme = self.themes[self.current_theme]
        
        # 设置表格样式
        self.packet_table.setStyleSheet(f""".QTableWidgetHeader {{
            background-color: {theme['table_header'].name()};
            color: {theme['text'].name()};
            font-weight: bold;
        }}""")
        
        # 设置文本框样式
        self.detail_text.setStyleSheet(f""".QTextEdit {{
            background-color: {theme['text_edit'].name()};
            color: {theme['text_edit_text'].name()};
        }}""")
        self.stats_text.setStyleSheet(f""".QTextEdit {{
            background-color: {theme['text_edit'].name()};
            color: {theme['text_edit_text'].name()};
        }}""")
        self.interface_text.setStyleSheet(f""".QTextEdit {{
            background-color: {theme['text_edit'].name()};
            color: {theme['text_edit_text'].name()};
        }}""")
    
    def toggle_theme(self, theme=None):
        """切换主题"""
        if theme:
            self.current_theme = theme
        else:
            self.current_theme = "dark" if self.current_theme == "light" else "light"
        
        # 更新主题动作的选中状态
        for name, action in self.theme_actions.items():
            action.setChecked(name == self.current_theme)
        
        self.apply_theme()
    
    def bind_shortcuts(self):
        """绑定快捷键"""
        # 已经在菜单中绑定了快捷键
        pass
    
    def update_interfaces(self):
        """更新可用网络接口列表"""
        from scapy.all import IFACES
        
        self.interface_combo.clear()
        self.interfaces = {}
        
        for name, iface in IFACES.items():
            friendly_name = f"{iface.description} ({name})"
            if hasattr(iface, 'ip') and iface.ip:
                friendly_name += f" - {iface.ip}"
            self.interface_combo.addItem(friendly_name)
            self.interfaces[friendly_name] = name
        
        self.status_bar.showMessage(f"已加载 {len(self.interfaces)} 个网络接口")
    
    def start_capture(self):
        """开始捕获数据包"""
        if self.is_capturing:
            QMessageBox.warning(self, "警告", "捕获已在进行中")
            return
        
        selected_friendly_name = self.interface_combo.currentText()
        if not selected_friendly_name:
            QMessageBox.critical(self, "错误", "请选择网络接口")
            return
        
        # 获取实际的接口名称
        interface = self.interfaces.get(selected_friendly_name, selected_friendly_name)
        filter_rule = self.filter_edit.text()
        
        # 开始捕获线程
        self.is_capturing = True
        self.status_bar.showMessage(f"正在捕获数据包...")
        
        # 更新按钮状态
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(True)
        
        # 清空之前的数据包
        self.clear_packets(clear_original=True)
        
        # 创建捕获线程
        self.capture_thread = CaptureThread(
            self.capture, interface, filter_rule
        )
        self.capture_thread.packet_captured.connect(self.on_packet_captured)
        self.capture_thread.capture_stopped.connect(self.on_capture_stopped)
        self.capture_thread.capture_error.connect(self.on_capture_error)
        self.capture_thread.start()
    
    def stop_capture(self):
        """停止捕获数据包"""
        if self.is_capturing:
            self.capture_thread.stop()
            self.is_capturing = False
            self.status_bar.showMessage(f"捕获已停止，共捕获 {self.packet_count} 个数据包")
            
            # 更新按钮状态
            self.start_capture_btn.setEnabled(True)
            self.stop_capture_btn.setEnabled(False)
        else:
            QMessageBox.information(self, "提示", "当前未在捕获数据包")
    
    def on_packet_captured(self, packet):
        """捕获到单个数据包时的处理"""
        # 将数据包添加到列表
        self.captured_packets.append(packet)
        self.original_packets.append(packet)  # 保存到原始数据包列表
        self.packet_count += 1
        
        # 识别协议
        protocol = self.identifier.identify_protocol(packet)
        
        # 获取基本信息
        source = ""
        destination = ""
        src_port = ""
        dst_port = ""
        length = len(packet)
        info = ""
        
        # 获取IP层信息
        from scapy.layers.inet import IP
        from scapy.layers.inet6 import IPv6
        from scapy.layers.inet import TCP, UDP
        
        if packet.haslayer(IP):
            ip = packet.getlayer(IP)
            source = ip.src
            destination = ip.dst
        elif packet.haslayer(IPv6):
            ipv6 = packet.getlayer(IPv6)
            source = ipv6.src
            destination = ipv6.dst
        
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
        self.add_packet_to_table(packet, protocol, source, destination, length, info)
        
        # 更新状态条
        self.status_bar.showMessage(f"正在捕获数据包... 已捕获 {self.packet_count} 个数据包")
    
    def add_packet_to_table(self, packet, protocol, source, destination, length, info):
        """将数据包添加到表格"""
        row = self.packet_table.rowCount()
        self.packet_table.insertRow(row)
        
        # 设置表格项
        items = [
            QTableWidgetItem(str(self.packet_count)),
            QTableWidgetItem("0.000000"),  # 时间暂时设为0，后续可以添加实际时间
            QTableWidgetItem(source),
            QTableWidgetItem(destination),
            QTableWidgetItem(protocol),
            QTableWidgetItem(str(length)),
            QTableWidgetItem(info)
        ]
        
        for i, item in enumerate(items):
            item.setTextAlignment(Qt.AlignCenter)
            self.packet_table.setItem(row, i, item)
    
    def on_capture_stopped(self):
        """捕获停止时的处理"""
        self.is_capturing = False
        self.status_bar.showMessage(f"捕获已停止，共捕获 {self.packet_count} 个数据包")
        
        # 更新按钮状态
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
    
    def on_capture_error(self, error):
        """捕获错误时的处理"""
        QMessageBox.critical(self, "捕获错误", f"捕获失败: {error}")
        self.status_bar.showMessage("捕获失败")
        
        # 更新按钮状态
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
        self.is_capturing = False
    
    def on_packet_selected(self):
        """数据包选中时的处理"""
        selected_items = self.packet_table.selectedItems()
        if not selected_items:
            return
        
        row = selected_items[0].row()
        if 0 <= row < len(self.captured_packets):
            packet = self.captured_packets[row]
            protocol = self.identifier.identify_protocol(packet)
            decoded_info = self.decoder.decode_packet(packet, protocol)
            
            # 显示详情
            self.detail_text.clear()
            self.detail_text.append(f"=== 数据包 {row+1} 详情 ===\n")
            self.detail_text.append(decoded_info)
    
    def open_pcap_file(self):
        """打开PCAP文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开PCAP文件", "", "PCAP文件 (*.pcap);;所有文件 (*.*)"
        )
        if file_path:
            try:
                self.status_bar.showMessage(f"正在读取PCAP文件...")
                packets = self.storage.read_pcap(file_path)
                # 先清空，再设置数据包列表
                self.clear_packets(clear_original=True)
                self.captured_packets = packets
                self.original_packets = packets.copy()  # 保存原始数据包
                
                # 显示数据包
                for packet in packets:
                    protocol = self.identifier.identify_protocol(packet)
                    
                    # 获取基本信息
                    source = ""
                    destination = ""
                    length = len(packet)
                    info = ""
                    
                    # 获取IP层信息
                    from scapy.layers.inet import IP
                    from scapy.layers.inet6 import IPv6
                    from scapy.layers.inet import TCP, UDP
                    
                    if packet.haslayer(IP):
                        ip = packet.getlayer(IP)
                        source = ip.src
                        destination = ip.dst
                    elif packet.haslayer(IPv6):
                        ipv6 = packet.getlayer(IPv6)
                        source = ipv6.src
                        destination = ipv6.dst
                    
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
                    
                    # 添加到表格
                    self.packet_count += 1
                    self.add_packet_to_table(packet, protocol, source, destination, length, info)
                
                self.status_bar.showMessage(f"已读取 {len(packets)} 个数据包")
            except Exception as e:
                QMessageBox.critical(self, "读取错误", f"读取PCAP文件失败: {str(e)}")
                self.status_bar.showMessage("读取失败")
    
    def save_pcap_file(self):
        """保存PCAP文件"""
        if not self.captured_packets:
            QMessageBox.warning(self, "警告", "没有可保存的数据包")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存PCAP文件", "", "PCAP文件 (*.pcap);;所有文件 (*.*)"
        )
        if file_path:
            try:
                self.status_bar.showMessage(f"正在保存PCAP文件...")
                if self.storage.save_pcap(file_path, self.captured_packets):
                    self.status_bar.showMessage(f"已保存 {len(self.captured_packets)} 个数据包")
                    QMessageBox.information(self, "成功", "PCAP文件保存成功")
                else:
                    raise Exception("保存失败")
            except Exception as e:
                QMessageBox.critical(self, "保存错误", f"保存PCAP文件失败: {str(e)}")
                self.status_bar.showMessage("保存失败")
    
    def clear_packets(self, clear_original=False):
        """清空数据包列表"""
        self.packet_table.setRowCount(0)
        self.detail_text.clear()
        self.captured_packets = []
        if clear_original:
            self.original_packets = []  # 仅在需要时清空原始数据包
        self.packet_count = 0
        self.status_bar.showMessage("数据包列表已清空")
    
    def search_packets(self):
        """搜索数据包"""
        search_text = self.search_edit.text().strip().lower()
        if not search_text:
            # 恢复显示所有数据包
            self.restore_all_packets()
            return
        
        # 过滤数据包
        filtered_packets = []
        for packet in self.original_packets:
            # 获取数据包的文本表示
            packet_text = str(packet)
            if search_text in packet_text.lower():
                filtered_packets.append(packet)
        
        # 更新数据包列表
        self.packet_table.setRowCount(0)
        self.captured_packets = filtered_packets
        
        # 显示过滤后的数据包
        for i, packet in enumerate(filtered_packets):
            protocol = self.identifier.identify_protocol(packet)
            
            # 获取基本信息
            source = ""
            destination = ""
            length = len(packet)
            info = ""
            
            # 获取IP层信息
            from scapy.layers.inet import IP
            from scapy.layers.inet6 import IPv6
            from scapy.layers.inet import TCP, UDP
            
            if packet.haslayer(IP):
                ip = packet.getlayer(IP)
                source = ip.src
                destination = ip.dst
            elif packet.haslayer(IPv6):
                ipv6 = packet.getlayer(IPv6)
                source = ipv6.src
                destination = ipv6.dst
            
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
            
            # 添加到表格
            self.add_packet_to_table(packet, protocol, source, destination, length, info)
        
        self.status_bar.showMessage(f"已搜索到 {len(filtered_packets)} 个数据包")
    
    def clear_search(self):
        """清除搜索条件"""
        self.search_edit.clear()
        self.restore_all_packets()
    
    def restore_all_packets(self):
        """恢复显示所有数据包"""
        self.packet_table.setRowCount(0)
        self.captured_packets = self.original_packets.copy()
        
        # 重新显示所有数据包
        for packet in self.captured_packets:
            protocol = self.identifier.identify_protocol(packet)
            
            # 获取基本信息
            source = ""
            destination = ""
            length = len(packet)
            info = ""
            
            # 获取IP层信息
            from scapy.layers.inet import IP
            from scapy.layers.inet6 import IPv6
            from scapy.layers.inet import TCP, UDP
            
            if packet.haslayer(IP):
                ip = packet.getlayer(IP)
                source = ip.src
                destination = ip.dst
            elif packet.haslayer(IPv6):
                ipv6 = packet.getlayer(IPv6)
                source = ipv6.src
                destination = ipv6.dst
            
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
            
            # 添加到表格
            self.packet_count += 1
            self.add_packet_to_table(packet, protocol, source, destination, length, info)
        
        self.status_bar.showMessage(f"已显示所有 {len(self.captured_packets)} 个数据包")
    
    def apply_filter(self):
        """应用过滤规则，对已捕获的包进行筛选"""
        filter_text = self.filter_edit.text().strip().lower()
        if not filter_text:
            QMessageBox.information(self, "提示", "请输入过滤规则")
            return
        
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
            from scapy.layers.inet6 import IPv6
            from scapy.layers.inet import TCP, UDP
            
            if packet.haslayer(IP):
                ip = packet.getlayer(IP)
                source = ip.src
                destination = ip.dst
            elif packet.haslayer(IPv6):
                ipv6 = packet.getlayer(IPv6)
                source = ipv6.src
                destination = ipv6.dst
            
            # 获取端口信息
            if packet.haslayer(TCP):
                tcp = packet.getlayer(TCP)
                src_port = str(tcp.sport)
                dst_port = str(tcp.dport)
            elif packet.haslayer(UDP):
                udp = packet.getlayer(UDP)
                src_port = str(udp.sport)
                dst_port = str(udp.dport)
            
            # 检查过滤条件
            if (
                filter_text in protocol or
                filter_text in source.lower() or
                filter_text in destination.lower() or
                filter_text in src_port or
                filter_text in dst_port
            ):
                filtered_packets.append(packet)
        
        # 更新数据包列表
        self.packet_table.setRowCount(0)
        self.captured_packets = filtered_packets
        self.packet_count = 0
        
        # 显示过滤后的数据包
        for packet in filtered_packets:
            protocol = self.identifier.identify_protocol(packet)
            
            # 获取基本信息
            source = ""
            destination = ""
            length = len(packet)
            info = ""
            
            # 获取IP层信息
            from scapy.layers.inet import IP
            from scapy.layers.inet6 import IPv6
            from scapy.layers.inet import TCP, UDP
            
            if packet.haslayer(IP):
                ip = packet.getlayer(IP)
                source = ip.src
                destination = ip.dst
            elif packet.haslayer(IPv6):
                ipv6 = packet.getlayer(IPv6)
                source = ipv6.src
                destination = ipv6.dst
            
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
            
            # 添加到表格
            self.packet_count += 1
            self.add_packet_to_table(packet, protocol, source, destination, length, info)
        
        self.status_bar.showMessage(f"已过滤出 {len(filtered_packets)} 个数据包")
    
    def clear_filter(self):
        """清除过滤规则，恢复显示所有数据包"""
        self.filter_edit.clear()
        self.restore_all_packets()
    
    def update_statistics(self):
        """更新统计信息，只包含文本统计"""
        if not self.captured_packets:
            self.stats_text.setText("暂无统计信息")
            return
        
        # 计算统计数据
        protocol_counts, source_counts, dest_counts = self._calculate_statistics()
        
        # 更新文本统计信息
        self._update_stats_text(protocol_counts, source_counts, dest_counts)
    
    def _calculate_statistics(self):
        """计算统计数据"""
        # 按协议统计
        protocol_counts = {}
        for packet in self.captured_packets:
            protocol = self.identifier.identify_protocol(packet)
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        
        # 按源IP统计
        source_counts = {}
        for packet in self.captured_packets:
            from scapy.layers.inet import IP
            from scapy.layers.inet6 import IPv6
            if packet.haslayer(IP):
                ip = packet.getlayer(IP)
                source_counts[ip.src] = source_counts.get(ip.src, 0) + 1
            elif packet.haslayer(IPv6):
                ipv6 = packet.getlayer(IPv6)
                source_counts[ipv6.src] = source_counts.get(ipv6.src, 0) + 1
        
        # 按目的IP统计
        dest_counts = {}
        for packet in self.captured_packets:
            from scapy.layers.inet import IP
            from scapy.layers.inet6 import IPv6
            if packet.haslayer(IP):
                ip = packet.getlayer(IP)
                dest_counts[ip.dst] = dest_counts.get(ip.dst, 0) + 1
            elif packet.haslayer(IPv6):
                ipv6 = packet.getlayer(IPv6)
                dest_counts[ipv6.dst] = dest_counts.get(ipv6.dst, 0) + 1
        
        return protocol_counts, source_counts, dest_counts
    
    def _update_stats_text(self, protocol_counts, source_counts, dest_counts):
        """更新文本统计信息"""
        stats = "数据包统计信息:\n\n"
        
        # 按协议统计
        stats += "按协议统计:\n"
        for protocol, count in protocol_counts.items():
            percentage = (count / len(self.captured_packets)) * 100
            stats += f"  {protocol}: {count} 个 ({percentage:.1f}%)\n"
        
        # 按源IP统计
        stats += "\n按源IP统计 (前10个):\n"
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for ip, count in sorted_sources:
            stats += f"  {ip}: {count} 个\n"
        
        # 按目的IP统计
        stats += "\n按目的IP统计 (前10个):\n"
        sorted_dests = sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for ip, count in sorted_dests:
            stats += f"  {ip}: {count} 个\n"
        
        self.stats_text.setText(stats)
    
    def update_interface_status(self):
        """更新接口状态"""
        from scapy.all import IFACES
        
        status_text = "网络接口状态和统计信息:\n\n"
        
        for name, iface in IFACES.items():
            status_text += f"接口名称: {name}\n"
            status_text += f"  描述: {iface.description}\n"
            status_text += f"  MAC地址: {iface.mac if hasattr(iface, 'mac') else '未知'}\n"
            status_text += f"  IP地址: {iface.ip if hasattr(iface, 'ip') else '未知'}\n"
            status_text += f"  索引: {iface.index if hasattr(iface, 'index') else '未知'}\n"
            status_text += f"  标志: {iface.flags if hasattr(iface, 'flags') else '未知'}\n\n"
        
        self.interface_text.setText(status_text)
    
    def show_about(self):
        """显示关于信息"""
        about_text = "协议分析软件 v1.0\n\n"
        about_text += "一款功能强大的网络协议分析工具，支持多种常见网络协议的识别、解码和存储。\n\n"
        about_text += "功能特性：\n"
        about_text += "- 协议识别：TCP/IP、UDP、HTTP、HTTPS、FTP、SMTP、POP3、IMAP等\n"
        about_text += "- 数据捕获：支持从网络接口捕获数据包\n"
        about_text += "- 过滤规则：支持BPF过滤语法\n"
        about_text += "- 协议解码：深入解析各层协议信息\n"
        about_text += "- 数据存储：支持PCAP格式文件的读写\n"
        about_text += "- 数据包筛选：支持按协议、IP地址、端口号等筛选\n\n"
        about_text += "开发语言：Python 3\n"
        about_text += "核心库：Scapy 2.5.0, PyQt5\n"
        about_text += "统计图表：Matplotlib\n"
        
        QMessageBox.information(self, "关于", about_text)
    
    def on_capture_stopped(self):
        """捕获停止时的处理"""
        self.is_capturing = False
        self.status_bar.showMessage(f"捕获已停止，共捕获 {self.packet_count} 个数据包")
        
        # 更新按钮状态
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
        
        # 更新统计信息
        self.update_statistics()
    
    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否有PCAP文件
            for url in event.mimeData().urls():
                if url.toLocalFile().endswith('.pcap'):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dragMoveEvent(self, event):
        """拖拽移动事件"""
        event.acceptProposedAction()
    
    def dropEvent(self, event):
        """拖拽释放事件，处理PCAP文件打开"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.endswith('.pcap'):
                    # 打开PCAP文件
                    self._open_dropped_pcap(file_path)
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def _open_dropped_pcap(self, file_path):
        """打开拖拽的PCAP文件"""
        try:
            self.status_bar.showMessage(f"正在读取PCAP文件...")
            packets = self.storage.read_pcap(file_path)
            # 先清空，再设置数据包列表
            self.clear_packets(clear_original=True)
            self.captured_packets = packets
            self.original_packets = packets.copy()  # 保存原始数据包
            
            # 显示数据包
            for packet in packets:
                protocol = self.identifier.identify_protocol(packet)
                
                # 获取基本信息
                source = ""
                destination = ""
                length = len(packet)
                info = ""
                
                # 获取IP层信息
                from scapy.layers.inet import IP
                from scapy.layers.inet6 import IPv6
                from scapy.layers.inet import TCP, UDP
                
                if packet.haslayer(IP):
                    ip = packet.getlayer(IP)
                    source = ip.src
                    destination = ip.dst
                elif packet.haslayer(IPv6):
                    ipv6 = packet.getlayer(IPv6)
                    source = ipv6.src
                    destination = ipv6.dst
                
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
                
                # 添加到表格
                self.packet_count += 1
                self.add_packet_to_table(packet, protocol, source, destination, length, info)
            
            self.status_bar.showMessage(f"已读取 {len(packets)} 个数据包")
        except Exception as e:
            QMessageBox.critical(self, "读取错误", f"读取PCAP文件失败: {str(e)}")
            self.status_bar.showMessage("读取失败")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion样式，支持主题切换
    window = ProtocolAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())
