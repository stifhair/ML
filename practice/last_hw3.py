import xml.etree.ElementTree as ET

data = '''
<person>
    <name>Chuck</name>
    <python type= "intl">
        *1 734 303 4456
    </python>
    <email hide="yes"/>
</person>
'''
tree = ET.fromstring (data) # parsing

lst = tree.findall('users/user')
print(lst)
for item in lst :
    print(item.find('name').text)
    print(item.find('id').text)
    print(item.get("x"))
print('Name: ', tree.find('name').text)
print('Attr ' , tree.find('email').get('hide'))