


from typing import List, Optional


class sommet:
    def __init__(self, Valeur ):
        self.Valeur = Valeur

class arrete:
    
    sommet1:sommet
    sommet2:sommet
    poids:Optional[int]
    
    def __init__(self, sommet1:sommet , sommet2:sommet , poids:Optional[int]=None ):
        self.sommet1 = sommet1
        self.sommet2 = sommet2
        self.poids = poids


'''
class arbre:
    
    racine:sommet
    sous_arbre:List[sommet]
    
    def __init__(self,racine:sommet,sous_arbre:List[sommet]):
        self.racine=racine
        self.sous_arbre=sous_arbre


'''
class arbre:
    
    racine:sommet
    sous_arbre:Optional['List[arbre]']
    
    def __init__(self, racine:sommet , sous_arbre:Optional['List[arbre]']=None ):
        self.racine=racine
        self.sous_arbre=sous_arbre

class arbre_binaire:
    
    racine:sommet
    gauche:Optional['arbre_binaire']
    droite:Optional['arbre_binaire']
    
    def __init__(self, racine:sommet , gauche:Optional['arbre_binaire']=None , droite:Optional['arbre_binaire']=None ):
        self.racine=racine
        self.gauche=gauche
        self.droite=droite

class graphe:
    
    sommets:List[sommet]
    arretes:List[arrete]
    
    def __init__(self, sommets:List[sommet] , arretes:List[arrete] ):
        self.sommets=sommets
        self.arretes=arretes

class graphe_cycle:
    
    sommets:List[sommet]
    
    def __init__(self, sommets:List[sommet] ):
        self.sommets=sommets






























s1=sommet(1)
s2=sommet(2)
s3=sommet(3)
s4=sommet(4)
s5=sommet(5)

arrete1_2=arrete(s1,s2)
arrete1_3=arrete(s1,s3)
arrete1_4=arrete(s1,s4)
arrete1_5=arrete(s1,s5)
arrete2_3=arrete(s2,s3)
arrete2_4=arrete(s2,s4)
arrete2_5=arrete(s2,s5)
arrete3_4=arrete(s3,s4)
arrete3_5=arrete(s3,s5)
arrete4_5=arrete(s4,s5)


sommets=[s1,s2,s3,s4,s5]
arretes=[arrete1_2,arrete1_3,arrete1_4,arrete1_5, arrete2_3,arrete2_4,arrete2_5, arrete3_4,arrete3_5, arrete4_5]

# Graphe complet K5
K5=graphe(sommets,arretes)


s6=sommet(6)

arrete1_6=arrete(s1,s6)
arrete2_6=arrete(s2,s6)
arrete3_6=arrete(s3,s6)


# Graphe biparti complet K3,3
K3_3=graphe([s1,s2,s3,s4,s5,s6],[arrete1_4,arrete1_5,arrete1_6, arrete2_4,arrete2_5,arrete2_6, arrete3_4,arrete3_5,arrete3_6])





