class PartyAnimal:
    x = 0
    name = "no name"

    def __init__(self,name):
        self.name = name
        print("I am constructed")

    def party(self):
        self.x = self.x + 1
        print(self.name, "So far", self.x)

    def __del__(self):
        print(self.name,"I am destructed")

class FootballFan(PartyAnimal):
    points = 0
    def touchdown(self):
        self.points = self.points + 7
        self.party()
        print (  self .name, "points", self.points)

a = PartyAnimal('a')
b = PartyAnimal('b')

a.party()
a.party()

b.party()
b.party()

j = FootballFan("Jim")
j.party()
j.touchdown()
