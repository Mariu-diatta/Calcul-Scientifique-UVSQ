#!/usr/bin/env python
# coding: utf-8

## Auteur: Marius DIATTA étudiant en M1 Algèbre Appliquée UVSQ

# # TP1

# # Exercice1

# In[3]:


from math import sqrt
from math import sqrt
import cmath


# Partie 1
def real_roots_quadra(a, b, c):
    # calcul de delta
    delta = b * b - 4 * a * c

    # evaluation de delta et calcul de racines

    # cas ou delta est positif, différent de 0:àn a deux racines
    if delta != 0 and delta > 0:
        sol1 = (-b - sqrt(delta)) / (2 * a)
        sol2 = (-b + sqrt(delta)) / (2 * a)
        return [sol1, sol2]

    # as ou delta est égual à 0 et le a différent de 0: on a une racine double
    if a != 0 and delta == 0:
        sol1 = -b / (2 * a)
        return sol1

    # cas ou delta est strictement inférieur à  0 et a non nul: deux racines complexe
    elif a != 0 and delta < 0:
        sol1 = (complex(-b, -sqrt(-delta)) / (2 * a))
        sol2 = (complex(-b, sqrt(-delta)) / (2 * a))
        return [sol1, sol2]

    # le cas ou a est égual à 0 l'equation devient  de la forme bx+c=0
    elif b != 0:
        sol1 = -c / b
        return sol1
    else:
        if c < 0:
            sol1 = sqrt(0, sqrt(-c))
            return [sol1]
        elif c > 0:
            sol1 = sqrt(c)
            return [sol1]
        else:
            return 0


# partie 2
def cplx_zeros_quadra(a, b, c):
    delta = b ** 2 - 4 * a * c

    # partie imaginaire et réel de Delta
    Im = delta.imag
    Re = delta.real

    r = sqrt((Re + sqrt(pow(Re, 2) + pow(Im, 2))) / 2) + sqrt((-Re + sqrt(pow(Re, 2) + pow(Im, 2))) / 2) * 1j

    Z1 = (-b - r) / (2 * a)
    Z2 = (-b + r) / (2 * a)

    return (Z1, Z2)


### (C) affichage des solutions avec module cmath

def ccplx_zeros_quadra(a, b, c):
    delta = b ** 2 - 4 * a * c

    # partie imaginaire et réel de Delta
    Im = delta.imag
    Re = delta.real

    r = cmath.sqrt((Re + cmath.sqrt(pow(Re, 2) + pow(Im, 2))) / 2) + cmath.sqrt(
        (-Re + sqrt(pow(Re, 2) + pow(Im, 2))) / 2) * 1j

    Z1 = (-b - r) / (2 * a)
    Z2 = (-b + r) / (2 * a)

    return (Z1, Z2)


# Tests des fonctions de l'exercice 1
# In[4]:


### (b) affichage des solutions

# cas Z^¨2 + Z + 1 = 0: a=b=c=1

cplx_zeros_quadra(1, 1, 1)

# In[6]:


#  résulta == ((-0.5-0.7071067811865476j), (-0.5-0.7071067811865476j))
# cas Z^¨2 -(3+4i)Z -2 +i = 0: a=1, b=-3 -4i, c = -2 + 6i
cplx_zeros_quadra(1, complex(-3, -4), complex(-2, 6))

# In[8]:


# cas Z^¨2 + Z + 1 = 0: a=b=c=1
ccplx_zeros_quadra(1, 1, 1)

# In[9]:


ccplx_zeros_quadra(1, complex(-3, -4), complex(-2, 6))


# # Exercice 2

# In[16]:


## question numéro 1

def pythagorrian(M):
    LesTrp = []  # liste des triples initialement vide

    # recherche de triplets
    for n in range(1, M + 1):
        for m in range(1, M + 1):
            for k in range(1, m + 1):
                if pow(k, 2) + pow(m, 2) == pow(n, 2):
                    LesTrp.append([k, m, n])  # ajout de triplet trouvé

    return LesTrp  # affichage de la liste contenant tous les triplets


## question numéro 2
def testVerification(n):
    if 2 <= n and 1000 >= n:

        ##on calculele carré de la sommes des nombre de 1 à n
        # on a une suite arithemetique de raison 1

        sommePremierDernierTerme = n + 1
        carreSomme = pow(n, 2) * pow(sommePremierDernierTerme, 2) / 4

        sommesDesCubes = 0

        # on calcule la sommes des cubes des nombre de 1 à n
        for i in range(1, n + 1):
            sommesDesCubes += pow(i, 3)

        if carreSomme == sommesDesCubes:
            return True
        else:
            return False
    else:
        print("Verifier cette condition: 2 <= n <= 1000 ")


##possibilité d'obtenir 100 euros
def possibilite():
    cpt = 0

    # pour cherché le nombre de possibilité, on cherche le nombre de combinaison possible de 2, 5 et 10 pour avoir 100
    # on divise donc 100 par 2, 5 et 10 pour avoir le nobre de fois qu'on peut combiner 2, 5 et 10
    for k in range(0, 11):
        for i in range(0, 21):
            for j in range(0, 51):
                if k * 10 + i * 5 + j * 2 == 100:
                    cpt += 1
                    posib = [k * 10, i * 5, j * 2]
                    print("la possibilité: {0}, numéro: {1}".format(posib, cpt))


# tests des fonctions de l'exercice 2
# In[17]:


pythagorrian(90)

# In[18]:


testVerification(100)

# In[19]:


possibilite()

# # Exercice 3

# In[20]:


# question 1
import matplotlib.pyplot as plt


def pop_simple_model(No, T, m):
    listeDesNi = [No]
    tmp = No

    for i in range(1, m + 1):
        N_suiv = (1 + T) * tmp
        tmp = N_suiv
        listeDesNi.append(N_suiv)

    return listeDesNi


# test fonction question 1
# In[21]:


# 3 Visualisation des Résultats
pop_simple_model(40, 1.7, 35)
t = [n for n in range(36)]
plt.plot(t, pop_simple_model(40, 1.7, 35))
plt.xlabel("abscisses")
plt.ylabel("ordonnées")
plt.show()


# In[22]:


# question 2
def pop_autolimitation_model(No, alpha, mu, m):
    listeDesNi = [No]
    tmp = No

    for i in range(1, m + 1):
        N_suiv = tmp + ((alpha - 1) - alpha * mu * tmp) * tmp
        tmp = N_suiv
        Tau = (alpha - 1) - alpha * mu * No
        listeDesNi.append(N_suiv)

    return listeDesNi


# Test de la fonction question 2
# In[23]:


pop_autolimitation_model(40, 2.7, 0.0063, 35)
t = [n for n in range(36)]
plt.plot(t, pop_autolimitation_model(40, 2.7, 0.0063, 35))
plt.xlabel("abscisses")
plt.ylabel("ordonnées")
plt.show()

# # Exercie 4

# In[36]:


import matplotlib.pyplot as plt


def mon_model_epedimique(i16, m, liste_des_Ci):
    table_alpha = [0.117, 0.110, 0.103, 0.096, 0.088, 0.081, 0.073, 0.066, 0.059, 0.051, 0.044,
                   0.037, 0.029, 0.022, 0.015, 0.007, 0.0]

    tableau_in = [0 for k in range(16)] + [i16, i16]
    # on met la valeur des ik dans un tableau sachant que cette valeur est nulle po
    # pour un indice inférieure ou égale à 15

    for n in range(18, m + 1):  # calcul de In pour un n supérieur ou égal à 18 donné
        sommek = 0
        sommeik = 0
        for k in range(n):
            sommek += tableau_in[k]
        for k in range(17):
            sommeik += table_alpha[k] * tableau_in[n - 2 - k]
        i_n = liste_des_Ci[n] * (1 - sommek) * sommeik
        tableau_in.append(i_n)

    return tableau_in


# Tets des différents cas
# In[37]:


# liste des abscisses
t = [k for k in range(197)]

# In[41]:


# cas 1 :
cnCas1 = [2 for k in range(197)]
modeCas1 = mon_model_epedimique(0.001, 196, cnCas1)
plt.plot(t, modeCas1)
plt.xlabel("abscisses")
plt.ylabel("ordonnées")
plt.show()

# In[39]:


# cas 2:
cnCas2 = [2 for k in range(56)] + [0.9 for d in range(56, 197)]
modeCas2 = mon_model_epedimique(0.001, 196, cnCas2)
plt.plot(t, modeCas2)
plt.xlabel("abscisses")
plt.ylabel("ordonnées")
plt.show()

# In[40]:


# cas 3:
cnCas3 = [2 for k in range(56)] + [0.9 for d in range(56, 96)] + [2 for j in range(96, 197)]
modeCas3 = mon_model_epedimique(0.001, 196, cnCas3)
plt.plot(t, modeCas3)
plt.xlabel("abscisses")
plt.ylabel("ordonnées")
plt.show()

# In[ ]:




