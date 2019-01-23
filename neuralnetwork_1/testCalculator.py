from neuralweb import calculateY_NN, chooseInputs, calculateY # <--- hier heb je alleen maar calculateY_NN van nodig


print("Dit is pong")
print("En nu ga ik het netwerk een paar keer gebruiken.")
print()
speeltLinks = False
p1points = 0
p2points = 0
streak = 0
Hstreak = 0
while True:
    if speeltLinks:
        print("Speler links...")
    else:
        print("Speler rechts...")
    inputs = chooseInputs()
    x1 = inputs[0]
    x2 = inputs[2]
    y1 = inputs[1]
    y2 = inputs[3]
    print("Ik ga 'm aanroepen met balletje van (",x1,",",y1,") naar (",x2,",",y2,")")
    output = calculateY_NN(x1, y1, x2, y2, continuousLearning = True, networkPlaysLeft = speeltLinks)  # <--- zo roep je 'm aan vanuit Pong!
    print("En het Nervje voorspelt dat ie uitkomt op y =",output)
    plekBal = (int)(calculateY(x1, y1, x2, y2, predictLeft = speeltLinks))
    print("Het wiskundig berekende antwoord is:", plekBal)
    print()
    if abs(plekBal - output)>100:
        print("De speler heeft gemist!")
        if speeltLinks:
            print("Speler rechts heeft een punt")
            p1points += 1
            print("p rechts: ")
            print(p1points)
            print("p links: ")
            print(p2points)
            print()
            streak = 0
            if(p1points == 10000):
                print("p rechts heeft gewonnen!!")
                print("hoogste score is")
                print(Hstreak)
                exit()
        else:
            print("Speler rechts heeft een punt")
            p2points += 1
            print("p rechts: ")
            print(p1points)
            print("p links: ")
            print(p2points)
            print()
            if(p2points == 10000):
                print("p links heeft gewonnen!!")
                print("hoogste score is")
                print(Hstreak)
                exit()
    else:
        streak += 1
        if streak >= Hstreak:
            Hstreak = streak
    speeltLinks = not speeltLinks
    