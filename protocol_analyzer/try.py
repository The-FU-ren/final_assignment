from os import name
from scapy.all import *
    

packet=sniff(
    #iface="6DD99391-C715-44AF-A918-582DC847D9B2",
    count=20)
print("已抓取接口 6DD99391-C715-44AF-A918-582DC847D9B2 的一个数据包：")
print(packet.summary())