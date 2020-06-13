class Animal:
    x = 0
    def party(self):
        self.x = self.x + 1
        print("So far",self.x)

an = Animal()

print (an)
print(an.party())
print(an.party())
print(an.party())
