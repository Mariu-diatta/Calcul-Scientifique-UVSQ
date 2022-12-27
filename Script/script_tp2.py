
# !/usr/bin/env python
# coding: utf-8

# In[ ]:


Auteur: Marius DIATTA étudiant en M1 Algèbre Appliquée UVSQ


# # Exercice 1

# In[279]:


# Exercie 1
from math import sqrt
import numpy as np
from numpy.linalg import inv

# question 1
def FactCholesky(A):
    # initialisation des variables A=np.a rray(A)
    n=len ( A)
    C=np. z eros((n,n))
    coef2={}
    info=1

    # Transposé de la matrice A =C
    for i in range(0,n):
        for j in range(0,n):
            C[i,j]=A [j , i]
            # je met les élément de la matrice dans une liste pour gérer les coords (i,j)
            coef2[(i+1,j + 1) ] = A[i , j]

            # Vérifier si la matrice est inversible
    for i in range(0,n):
        for j in range(0,n):
            if C[i,j]!= A[ i, j]:
                info=0
                break

    # 0n décompose la matrice en cholesky et on renvoie sa décomposition

    # On vérifie si la matrice est bien inversible
    if info==0:
        C=np. z eros((n,n
            )) # on  remet tous les éléments de la matrice à zéros et on le renvoi car la matrice à decomposer
        # n'est pas inversible
        return (info, C)  # re  urne de info=0 si la matrice n'est pas inversible
    else:
        C=np.z e ros((n, n))
        for i in range(1,n+1):
            for j in range(1,n+1):
                if i==j a nd j==1:
                    C[(0,0)]=s qrt ( coef2[(1,1)])
                if i==1 a nd 1<j an d j<=n:
                    C[(j-1,0) ] =c oef 2 [(1,j)]/C [(0 , 0)]

                if 1<i an d i<=n a nd j==i:
                    som=sum( [ pow(C[(i-1,k- 1 )] , 2 ) fo r k in range(1,i)])
                    C[(i-1,i- 1 )] = s qrt ( coef2[(i,i)]-s om)

                if 1<i an d i<=n a nd i<j an d j<=n:
                    som1=sum( [ C[(i-1,k- 1 )] * C [(j - 1,k- 1 )] fo r k in range(1,i)])
                    C[(j-1,i-1 ) ]= ( c oef 2 [(i,j)]-so m1) / C[(i- 1 ,i-1 ) ]
    return (info, C)  # returne de info=1 si la matrice est inversible


# te


s Exercice 1
# In[280]:


# test pour la matrice A
A=np.ar r ay([[1,-2, 0] ,[-2,8, -6] , [0, -6, 25]])
M=FactC h olesky(A)
M


# I

81]:


# produit de la matrice M[1] et de sa transposé pour preuve de justification
print(np.dot(M[1],(M[1]. T)))


# I

05]:


# Test pour la matrice B
B=np.ar r ay([[4,0, 12, 6],[0,12, 2,1 ],[ 12 ,2, 49,- 4] ,[- 6,1, -4,5 1] ]) M1=FactCh o lesky(B)
M1


# # Exercice 2
#
# #importation de package
# import numpy as np

# In[131]:


# Exercice 2 question 1
def syst_Trinf(B,L):

    Det=1 info=T rue
    n= l en(L) Y=np.z eros(len(B))
    InvL=np.zeros((n,n))
    L=np.arra y(L) coef={ }
    coefB={} for i in range(0,le n (L)):
        Det*=L[i,i]

    for i in range( 0, len( B)):
        coefB[i+1]=B [i]

    for i in range( 0 ,l e n(L)):
        for j in ran ge(0,len(L)):
            coef[(i+ 1,j+1)]=L[i,j]

    if Det>0 :
        Y[0 ]=coefB[1]/coe f [(1,1)] for i in range(2, l en(B)+1) :
            som=sum([coef[( i,k)]* Y [k-1] for k in rang e (1, i)]) Y[i-1]=(1/coef[(i,i)])*(coefB[i]-som)
        return ( i nf o ,Y )

    el se:
        in f o=False
        return (in fo,Y)


# In[132]:


# tes t  de la question 1
syst_Trin f(np.array([1,2]),np.array([[3,0],[2,1]]))


# In[149]:


# Exerc ice  2 question 2 def syst _Trsup(B,U):
    Det=1
    info=True
    n=len(U)
    Y=np.zer os(len(B))
    U=np.ar r ay(U) coef={ }
    coefB= { }


    for i in range ( 0,n):
        Det*=U [ i,i]

    fo r i in \

    range(0,n):
        coefB[i+1 ]= B[i]

    for i in range(0,le n(U)):
        for j in ra nge(0,len(U)): coef[(i+1,j+1)]=U[i,j]

    if Det>0:
        Y [n-1]=coefB[n]/coef[(n,n)]
        f o r i in range(n-1,0, - 1):
            s o m=sum([c o ef[(i,k) ]*Y[k-1] for k in range(i+1, n+1 )] )
            Y[i-1] = (coefB[i]-som )/c o ef[ ( i,i)]

        retu r n (i n fo,Y)

    else:
        info=Fal s e
        ret urn (info,Y)


    # In[15 0]:


syst_Trsup(np.array( [ 1,2]),np.array([[1,2],[0,1] ]))

    # In[164]:


# Exercice 2 question  3
de f syst_LU(B,L ,U) :
    U=np.array(U)
    L=np.array(L)
    # résoudre LUX=B rev ie nt à réso u dre successivemen t  LX=B puis UY=X
    # retourn un tuble comportant un boolean et la solution de LX=B
    Y=syst_Trinf(B,L)
    # retourn un tuble comportant un boolean et l a  solution de  UY=X
    Z=syst_Trsup(Y[1],U)
    # Z est donc la solution de LUX=B
    retu r n Z


# In[165]:


# test question 3
syst_LU([1,2],np.array([[3,0],[2,1]]),np.array([[1,2],[0,1]]))


# # Exercice  3 Programmation # In [11 ]: import numpy as np


#  In[12]:


# (a) Ecriture d'une fonction mineurs

def supprimer_Coll_Lign(A,k):
    n=len(A)
    rg=range(n-k)
    B= [[None for p in rg] for q in r g]
    fo r q in rg:
        for p in rg:
            B[q][p]=A[q][p]
    return B

def mineurs(matrice_trign):
    A=np.array(matrice _ trign)
    n=len(mat


ce_trign)
    pl=[round(np.linalg . det(np.array(supprimer_Coll_L i gn(A,k))))  for k in rang e (n)]
    return pl[::-1]


# a = np.array(([-1,2],[ -3,4]))
# np.linalg.det(a)
A =np.array([[1,2,0],[


3,4,1],[0,3,5]])
mineurs(A)


# d=supprimer_Coll_Lig n(A,1)
# d # In[ 250 ]: # (b)  dec om position en LU


def decLU(U):
    B=np.diag(U)  # diagonale principale
    C=np.diag(U, k=1)# diagonale en-dess o us
    A=n  p.diag(U, k=-1)# diagonale au - dessus
    n=le  n(B)
    listeU=dict()
    l i steL=dict()  listeU[1]=B[0]

    # calcul du  d éterminant en uti l isant les diagona l es
    for i in rang e (2,len(B)+1):
        listeL[i] = A[i-2]/listeU[i-1]
        listeU[i]= B[i-1]-list eL[i]* C [i-2]

    U=np.diag([liste U [q ] for q i n listeU])+np.diag(C, k=1)#   di a gonale en - des s ous de la   la diagonale prinicipale de U
    L= n p.diag([li
                                                       steL  [q] for q in listeL], k=-1)+np.identity(len(B))# diagonale au-de s sus de la la diagonale prinicipale de U
    print(np.do
        t(L,U))
    return [U,L]


# In[251]:


# test de la fonction


# In[249]:


# const ruction de la matr ice de test
B=[2 for k in range(1,7)]
C=[-1 for k in range(1,6)]
A=[-1 for k in range(1,6)]
m a trix = np.diag(B)+np .diag ( C,k=1)+np.diag(A,k=-1 )
# t e st
decLU(matrix)


#  In[256]:


# (c) résolu t ion de d'é quat i on AX=B
de f mjacobi(A):
    Y=syst_Trinf(B,decLU(A)[0])
    Z=syst_Trsup(Y[1],decLU(A)[1])
    return Z


# Te s t
# In[257]:


mjacobi(matrix)


# In[ ]: # # Exercice 4

# In[ ]:




