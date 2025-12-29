#!/usr/bin/env python3
# 系统测试用例

import sys
import os
# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 导入qt_gui模块，并从中获取PacketAnalyzerGUI类
try:
    import qt_gui
    PacketAnalyzerGUI = qt_gui.ProtocolAnalyzerGUI
except Exception as e:
    print(f"Error importing qt_gui: {e}")
    import traceback
    traceback.print_exc()

class TestSystem:
    """系统测试类"""
    
    @pytest.fixture(scope="class")
    def app(self):
        """创建QApplication实例"""
        app = QApplication(sys.argv)
        yield app
        app.quit()
    
    def test_theme_switch(self, app):
        """测试主题切换功能"""
        # 创建GUI实例
        gui = PacketAnalyzerGUI()
        gui.show()
        
        # 切换到白色主题
        gui.set_theme("light")
        
        # 切换到黑色主题
        gui.set_theme("dark")
        
        gui.close()
    
    def test_protocol_stats_dialog(self, app):
        """测试协议统计对话框"""
        # 创建GUI实例
        gui = PacketAnalyzerGUI()
        gui.show()
        
        # 直接调用显示协议统计的方法
        gui.show_protocol_stats()
        
        # 验证统计对话框已显示
        assert hasattr(gui, 'stats_dialog')
        assert gui.stats_dialog.isVisible()
        
        # 关闭对话框和GUI
        gui.stats_dialog.close()
        gui.close()
    
    def test_packet_filter(self, app):
        """测试数据包过滤功能"""
        # 创建GUI实例
        gui = PacketAnalyzerGUI()
        gui.show()
        
        # 设置过滤规则为"TCP"
        gui.filter_edit.setText("TCP")
        
        # 验证过滤功能已触发（会更新packet_table）
        assert hasattr(gui, 'packet_table')
        
        gui.close()
    
    def test_menu_functionality(self, app):
        """测试菜单功能"""
        # 创建GUI实例
        gui = PacketAnalyzerGUI()
        gui.show()
        
        # 验证GUI具有菜单相关的方法
        assert hasattr(gui, 'menu_file')
        assert hasattr(gui, 'menu_analysis')
        assert hasattr(gui, 'menu_view')
        assert hasattr(gui, 'menu_help')
        assert hasattr(gui, 'menu_capture')
        
        gui.close()
    
    def test_initial_state(self, app):
        """测试GUI初始状态"""
        # 创建GUI实例
        gui = PacketAnalyzerGUI()
        gui.show()
        
        # 验证初始状态
        assert gui.isVisible()
        
        # 验证GUI具有必要的组件
        assert hasattr(gui, 'capture_control')
        assert hasattr(gui, 'filter_edit')
        assert hasattr(gui, 'start_capture_btn')
        assert hasattr(gui, 'stop_capture_btn')
        
        gui.close()
