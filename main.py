import csv, time, os, sys
from scapy.all import sniff, wrpcap, IP, IPv6, TCP, UDP, conf

OS = os.name

iface = conf.iface


f = open("packets.csv", "a", newline="")
w = csv.writer(f)
""" This code saves the data into a csv file """
def handle(packet):
    ts = time.time()
    ipver = None
    src = dst = proto = sport = dport = tcp_flags = None
    length = len(packet)
    if IP in packet:
        ipver = 4
        src, dst = packet[IP].src, packet[IP].dst
        proto = packet[IP].proto
    elif IPv6 in packet:
        ipver = 6
        src, dst = packet[IPv6].src, packet[IPv6].dst
        proto = packet[IPv6].nh

    if TCP in packet:
        sport, dport = packet[TCP].sport, packet[TCP].dport
        tcp_flags = int(packet[TCP].flags)
    elif UDP in packet:
        sport, dport = packet[UDP].sport, packet[UDP].dport

    w.writerow([ts, ipver, src, dst, proto, sport, dport, length, tcp_flags])
    f.flush()


sniff(prn=handle, store=False, filter="ip or ip6")

    
#stores 20 packets
packets = sniff(count=20, store = True)
wrpcap("PcapTest.pcap", packets)




#prints out the summary of captured
packets.nsummary()