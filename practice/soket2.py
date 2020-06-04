# coding:utf-8#

import socket
try:
    url = input('Enter : ')
    words = url.split('/')
    print(words)
    if words[0] != 'http:':
        hosts = words[0]
        url = 'http://' + url
    else:
        hosts = words[2]

    mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysock.connect((hosts, 80))
    mysock.send(('GET'+url+'HTTP/1.0\r\n\r\n').encode())
    n=''

    while True:
        data = mysock.recv(512)
        if len(data) < 1:
            break
        n=n+(data.decode(0))
        print(data.decode(),end='')
    print(n[:3001])
    print("Total number")
    mysock.close()
except:
    print("not formatted properly")


