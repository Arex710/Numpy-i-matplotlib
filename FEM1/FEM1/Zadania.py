import numpy as np
import matplotlib.pyplot as plt
import Z15

# 6 Ćwiczenia
#6.3

A1 = np.array([np.linspace(1,5,5), 
              np.linspace(5,1,5)])
A2 = np.zeros((3,2))
A3 = np.ones((2,3))*2
A4 = np.linspace(-90,-70,3)
A5 = np.ones((5,1))*10
#print(A1,"\n")

A = np.block([[A3], [A4]])
A = np.block([A2,A])
A = np.block([[A1],[A]])
A = np.block([A,A5])
print("Macierz A")
print(A)

#6.4
B = A[1] + A[3]
print("Macierz B")
print(B)

#6.5
C = list(map(np.max, zip(*A)))
print("Macierz C")
print(C)

#6.6
D = np.delete(B,0)
D = np.delete(D, len(D)-1)
print("Macierz D")
print(D)

#6.7
D[D==4] = 0
print("Macierz D")
print(D)

#6.8
E = np.delete(C, np.where(C == np.min(C)))
E = np.delete(E, np.where(E == np.max(E)))
print("Macierz E")
print(E)

#6.9
print("Wiersze z najwieksza wartoscia: ",A[np.where(A == np.max(A))[0]])
print("Wiersz z najmniejsza wartoscia: ",A[np.where(A == np.min(A))[0]])

#6.10
print("Mnożenie tablicowe: ", D * E)
print("Mnożenie wektorowe: ", D @ E)


#6.11
def function611(m):
    matrix = np.random.randint(0, 11, [m, m])
    return matrix, np.trace(matrix)

[Matrix, t] = function611(5)

print("Funkcja z zadania 11: \n ", Matrix)
print("Slad funkcji: ", t)

#6.12

def function612(matrix):
    if np.shape(matrix)[0]==np.shape(matrix)[1]:
        matrix = matrix * (1-np.eye(np.shape(matrix)[0], np.shape(matrix)[0]))
        matrix = matrix * (1-np.fliplr(np.eye(np.shape(matrix)[0], np.shape(matrix)[0])))
        return matrix
    else:
        return "Macierz nie jest kwadratowa"

print("Funkcja z zadania 12: \n",function612(Matrix))

#6.13

def fun613(matrix):
    if np.shape(matrix)[0]==np.shape(matrix)[1]:
        sum=0
        for i in range(1, np.shape(matrix)[0], 2):
            sum=sum+np.sum(matrix[i, :])
        return sum
    else:
        return "Macierz nie jest kwadratowa"

print("Funkcja z zadania 13: \n", fun613(Matrix))

#6.14
y1 = lambda x: np.cos(2*x)
x = np.linspace(-10, 10, 100)
plt.plot(x, y1(x), color='red', dashes=[2, 2])
#plt.show()

#6.15
plt.plot(x, Z15.y2(x), '+g')
plt.show()

#6.17
#print("\n#Zadanie 17\n")
#plt.plot(x, 3 * y1(x) + arr, '*b')
#plt.show()