import sqlite3

conn = sqlite3.connect('users.sqlite')
cur = conn.cursor()

#cur.execute('CREATE TABLE Users (name VARCHAR(128), email VARCHAR(128))')

cur.execute("DELETE FROM Users") #모든 것을 지운다는 것

data = [('Chuck', 'csev@umich.edu'), ('Colleen', 'cvl@umich.edu'), ('Ted', 'ted@umich.edu'), ('Sally', 'a1@umich.edu')]

for (name, email) in data:
    cur.execute("INSERT INTO Users (name, email) VALUES (?, ?)", (name, email))

conn.commit()

cur.execute("UPDATE Users SET name='Charles' WHERE email='csev@umich.edu'")
cur.execute("UPDATE Users SET name='Jiwon' WHERE email='cvl@umich.edu'")

conn.commit()

cur.execute("SELECT email FROM Users WHERE email='csev@umich.edu'")

row = cur.fetchall()
print(row)

conn.close()
