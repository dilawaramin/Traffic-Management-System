
while True:
    n = input("numerator")
    try:
        n = int(n)
        break
    except ValueError:
        print("input numbers please ")