import urllib.request

try:
    url = input('Enter : ')
    n = urllib.request.urlopen(url).read().decode()
    print(n[:3001])
    print('Total number:',len(n))
except:
    print("wrong format")
