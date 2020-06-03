# coding:utf-8#

import socket

url = input ("Enter:")

try:
    words = url.split('/')
    hosts = words[2]
    mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysock.connect((hosts, 80))
    mysock.send(('GET'+url+'HTTP/1.0\r\n\r\n').encode())
    # cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\r\n\r\n'.encode()
    # mysock.send(cmd)

    while True:
        data = mysock.recv(512)
        if len(data) < 1:
            break
        print(data.decode(),end='')
    mysock.close()
except:
    print("not formatted properly")
