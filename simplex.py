import numpy as np

# Funcion para leer los problemas del archivo
def parse_problems(filename):
    problems = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Detecta el inicio de un nuevo problema
        if line.isdigit():
            problem_number = int(line)
            problems[problem_number] = {'c': [], 'A': [], 'b': []}
            i += 1

            # Leer c
            if lines[i].strip().startswith('c='):
                i += 1
                while lines[i].strip() and not lines[i].strip().startswith('A='):
                    problems[problem_number]['c'].extend(map(int, lines[i].split()))
                    i += 1
            
            # Leer A
            if lines[i].strip().startswith('A='):
                i += 1
                while lines[i].strip() and not lines[i].strip().startswith('b='):
                    problems[problem_number]['A'].append(list(map(int, lines[i].split())))
                    i += 1
            
            # Leer b
            if lines[i].strip().startswith('b='):
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                    problems[problem_number]['b'].extend(map(int, lines[i].split()))
                    i += 1
        else:
            i += 1
    
    return problems


# Definición de la clase Simplex
class Simplex():
    def __init__(self,A,b,c,conj_B=None):
        # Guardar los datos del problema
        self.A=A
        self.b=b
        self.c=c
        self.m,self.n=A.shape
        self.conj_B=conj_B

    def fase_I(self):
        # Datos del problema aritificial
        c_artificial = np.hstack((np.zeros(self.n), np.ones(self.m)))
        A_artificial = np.hstack((self.A, np.eye(self.m)))
        conj_B_artificial=[i+len(self.c)+1 for i in range(self.m)]
        B_inv = np.eye(self.m)
        # Inicializar i resolver el problema artificial
        problema_artificial=Simplex(A_artificial,self.b,c_artificial,conj_B_artificial)
        res=problema_artificial.fase_II(B_inv)
        return res

    def fase_II(self,B_inv,iteracio=0):
        # Inicializar variables
        conj_N=[i for i in range(1,self.n+1) if i not in self.conj_B]
        An = [[fila[i-1] for i in conj_N] for fila in self.A]     
        cb = [self.c[i-1] for i in self.conj_B]
        cn = [self.c[i-1] for i in conj_N]
        self.xb = np.dot(B_inv,self.b)
        self.z=np.dot(cb,self.xb)

        # Bucle principal del algoritmo simplex
        while True:
            iteracio+=1
            print(f"\nIteración {iteracio}:   ",end='')
            print(f"z={np.round(self.z,2):.2f}", end='    ') 
            
            # Paso 1: Comprobar si es óptimo
            self.r=cn - np.dot(np.dot(cb,B_inv),An)

            if np.all(self.r>=0):
                return (self.conj_B,self.z,B_inv,iteracio) 
            
            # Regla de Bland
            for i in range(len(self.r)):
                if self.r[i]<0:
                    q=conj_N[i]
                    break

            # Paso 2: DBF
            Aq=[fila[q-1] for fila in self.A]
            db=-np.dot(B_inv,Aq)

            if all(db>=0):
                print("\n=== Problema no acotado ===")
                return None

            # Paso 3: Theta
            llista_theta=[-self.xb[i]/db[i] if db[i]<0 else np.inf for i in range(self.m)]
            theta=min(llista_theta) # Regla de Bland
            print(f"theta={np.round(theta,2):.2f}", end='    ')
            p=llista_theta.index(theta)

            # Paso 5: Cambio de base
            self.xb=self.xb+theta*db
            self.xb[p]=theta
            self.z=self.z+theta*self.r[conj_N.index(q)]
            if np.all(self.xb<0):
                print("\nError")
                return None
            

            indice_q=conj_N.index(q)
            self.conj_B[p],conj_N[indice_q]=conj_N[indice_q],self.conj_B[p]

            # Actualizar la inversa de B      
            matriu_E = np.eye(self.m)
            columna_P = np.zeros((self.m, 1))
            for i in range(self.m):
                if i == p:
                    columna_P[i,0] = -1/db[p]
                else:
                    columna_P[i,0] = -db[i]/db[p]
            matriu_E[:,p] = columna_P[:,0]
            B_inv = np.dot(matriu_E,B_inv)


            An = [[fila[i-1] for i in conj_N] for fila in self.A]     
            cb = [self.c[i-1] for i in self.conj_B]
            cn = [self.c[i-1] for i in conj_N]
            print(f"p={p}", end='    ')
            print(f"q={q}", end='    ')
            print(f"B[p]={conj_N[indice_q]}", end='')


    def resolver(self):
        print("=== Algoritmo Simplex ===")
        print("=== Fase I ===", end='')
        resultado=self.fase_I()
        if resultado:
            self.conj_B,z,B_inv,iteraciones_faseI=resultado
            if z> 0.001: # tolerancia para errores de coma flotante
                print("\n=== Problema infactible ===")
                return None
            print("\nSolucion básica factible encontrada")
            print("=== Fase II ===", end='')
            solucion=self.fase_II(B_inv,iteraciones_faseI)
            if solucion:
                self.conj_B,z,_,iteraciones=solucion
                print("\nSolución óptima encontrada")
                print("Fin del algoritmo")
                print()
                print("=== Resultados ===")
                print(f'Variables Basicas: {self.conj_B}')
                print(f'XB:                {np.round(self.xb,2)}')
                print(f'Valor óptimo:      {np.round(self.z,2)}')
                print(f'r:                 {np.round(self.r,2)}')


            return solucion


# Leer los problemas del archivo
filename = "input.txt" 
problemas = parse_problems(filename)

# Problemas
for i in range(1,9):
    print(f"\nProblema {i}\n")
    problema=i
    problema=Simplex(np.array(problemas[problema]['A']),np.array(problemas[problema]['b']),np.array(problemas[problema]['c']))
    solucion=problema.resolver()
    print("========================================================================")