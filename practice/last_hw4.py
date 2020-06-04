import xml.etree.ElementTree as ET

data = '''
<stuff>
    <users>
        <user x="2">
            <id>001</id>
            <name>Chuck</name>
        </user>
        <user x= "7">
            <id>009</id>
            <name>Brent</name>
        </user>
    </users>
</stuff>
'''
tree = ET.fromstring (data) # parsing

lst = tree.findall('users/user')
print(lst)
for item in lst :
    print(item.find('name').text)
    print(item.find('id').text)
    print(item.get("x"))
