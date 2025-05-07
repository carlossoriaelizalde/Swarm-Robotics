"""
Prueba Matriz y Robots buscando la comida y llevÃ¡ndola a su casa

    Los robots se mueven por la matriz, de forma que no se pueden salir de ella, y se mueven de 
    forma asÃ­ncrona para evitar que se coman. AdemÃ¡s, hay un semÃ¡foro para que hagan los cambios de uno en uno.
    
    Ahora los robots buscan comida y la llevan a casa

@author: carlo
"""

# from tqdm import tqdm  # Asegurate de tener instalada la libreri­a tqdm
import simpy
import numpy as np
import numpy.random as nprandom
import simpy.rt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
import random
import plotly.graph_objects as go
import webbrowser
import os

# NOTACION: EL MAPA LO CONSIDERO ASI

#    0  1  2  3  4
# 0
# 1
# 2
# 3
# 4

class FinalizarSimulacion(Exception):
    pass


class Mapa(object):
    def __init__(self, size, distancia_minima):
        self.matriz = np.zeros((size, size))
        self.posiciones = dict()
        self.size = size
        
        # Pongo la casa y la fuente de comida
        self.comida = nprandom.randint(0, self.size, 2)
        self.matriz[self.comida[0], self.comida[1]] = 2
        self.posiciones["Comida"] = self.comida
        #print("Food is in: ",self.comida)
        
        # Procuro que la casa este alejada (al menos distancia_minima de lejos de la comida)
        pos = nprandom.randint(0, self.size, 2)
        while np.linalg.norm(pos - self.comida) < distancia_minima:
            pos = nprandom.randint(0, self.size, 2)
        self.casa = pos
        self.matriz[pos[0], pos[1]] = 3
        self.posiciones["Casa"] = self.casa
        #print("Home is in: ",self.casa)
        
        # Creo un diccionario que guarda los robots sabios (True/False)
        self.robots_sabios = dict()
        self.robots_pista = dict()
        self.comidas_llevadas = dict()
        
        # Creo una lista donde ire guardando las graficas que vaya creando para ensenyarlas despues
        self.lista_graficas = []
        
        # Creo un booleano para que todos los robots paren
        self.parar_simulacion = False
    
    def inicializar_robot(self, name):
        pos = nprandom.randint(0, self.size, 2)
        # Procuro que los robots estÃ©n alejados de todos los objetos ya puestos
        while any(np.array_equal(pos, valor) for valor in self.posiciones.values()):
            pos = nprandom.randint(0, self.size, 2)
        self.posiciones[name] = pos
        self.matriz[pos[0], pos[1]] = 1
        self.robots_sabios[name] = False
        self.robots_pista[name] = False
        return pos

class Robot(object):
    def __init__(self, env, name, mapa, semaforo_cambio, num_robots, prob_inicial_pista, duracion_pista):
        self.env = env
        self.action = env.process(self.run())
        self.name = name
        self.mapa = mapa
        self.semaforo_cambio = semaforo_cambio
        self.num_robots = num_robots
        
        
        self.prob_inicial_pista = prob_inicial_pista
        self.duracion_pista = duracion_pista
        
        self.prob_maximas = []
        for i in range(1, self.duracion_pista+1):
            self.prob_maximas.append(0.25 + (self.prob_inicial_pista-0.25)/i)
        #print(self.prob_maximas)
        
        # Ponemos al robot en su posicion inicial
        self.posicion = self.mapa.inicializar_robot(self.name)
        #print(self.name, " starts in ", self.posicion)
        
        self.direcciones_naive = [[np.array([-1, 0]), 0.25], [np.array([1, 0]), 0.25], [np.array([0, -1]), 0.25], [np.array([0, 1]), 0.25]]
        self.direcciones_sabio = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]
        
        self.sabio = False
        self.lleva_comida = False
        
        self.direccion_pista = None
        self.iteracion_pista = -1
    
    # Funcion para ordenar la lista naive segun las probabilidades
    def revolver_ponderado(self, lista):
        # Separar los elementos con peso 0 del resto
        lista_copia = [item for item in lista if item[1] != 0]
        elementos_cero = [item for item in lista if item[1] == 0]
    
        # Lista para almacenar el resultado
        resultado = []
        while lista_copia:
            # Extraer probabilidades de la lista restante
            pesos = [item[1] for item in lista_copia]
            # Seleccionar un Ã­ndice basado en los pesos
            indice_seleccionado = random.choices(range(len(lista_copia)), weights=pesos, k=1)[0]
            # Agregar el elemento seleccionado al resultado
            resultado.append(lista_copia.pop(indice_seleccionado))
    
        # Agregar los elementos con peso 0 al final
        resultado.extend(elementos_cero)
        return resultado
    
    def calcular_direccion_comida(self, vector):
        # Crear la lista de vectores en base canonica
        base_canonica = []
        for i in range(len(vector)):
            # Crear un vector en base canonica
            canonico = np.zeros(len(vector))
            if vector[i] > 1:
                canonico[i] = 1  # Si el valor es mayor a 1, ajustar a 1
            elif vector[i] < 0:
                canonico[i] = -1  # Si el valor es negativo, ajustar a -1
            elif vector[i] > 0:
                canonico[i] = vector[i]  # Si esta entre 0 y 1, usar el valor original
            base_canonica.append(canonico)

        if np.any(vector == 0):  # Si hay un cero
            base_canonica = [base_canonica[np.argmax(np.abs(vector))]]  # Toma el mayor valor absoluto
        
        nprandom.shuffle(base_canonica)
        
        return base_canonica[0]
    
    def calcular_probabilidades(self):
        # Encontrar la sublista
        resultado = next((sublista for sublista in self.direcciones_naive if np.array_equal(sublista[0], self.direccion_pista)), None)
        
        # Asignarle a ese elemento la probabilidad que le toca
        resultado[1] = self.prob_maximas[self.iteracion_pista]
        
        # Encontrar la sublista contraria
        contrario = next((sublista for sublista in self.direcciones_naive if np.array_equal(-sublista[0], self.direccion_pista)), None)
        # Ponerle probabilidad cero
        contrario[1] = 0
        
        # Encontrar las otras sublistas
        otras = [sublista for sublista in self.direcciones_naive if not (np.array_equal(sublista[0], resultado[0]) or np.array_equal(sublista[0], contrario[0]))]
        
        for minilista in otras:
            minilista[1] = (1-self.prob_maximas[self.iteracion_pista])/2
    """        
    def graficar(self):
        fig, ax = plt.subplots()
        ax.clear()
        colors = {0: 'white', 1: 'black', 2: 'green', 3: 'red', 98: "purple", 99: 'blue'}
        for x in range(self.mapa.size):
            for y in range(self.mapa.size):
                value = self.mapa.matriz[x, y]
                # Si es o casilla vacia, o comida, o casa, lo pintamos tal cual
                if value in [0, 2, 3]:
                    ax.add_patch(plt.Rectangle((y, x), 1, 1, color=colors[value]))
                
                # Si no, hay un robot, pero cual de los tres tipos es?
                else:
                    robot_key = next((k for k, v in self.mapa.posiciones.items() if np.array_equal(v, [x, y])), None)
                    
                    # Si es sabio
                    if self.mapa.robots_sabios[robot_key]:
                        
                        ax.add_patch(plt.Rectangle((y, x), 1, 1, color=colors[99]))
                    
                    # Si tiene pista
                    elif self.mapa.robots_pista[robot_key]:
                        
                        ax.add_patch(plt.Rectangle((y, x), 1, 1, color=colors[98]))
                    
                    # Si no es ninguna de las anteriores es naive sin pista
                    else:
                        
                        ax.add_patch(plt.Rectangle((y, x), 1, 1, color=colors[1]))
                     
        
        ax.set_xlim(0, self.mapa.size)
        ax.set_ylim(0, self.mapa.size)
        
        # Configurar rejilla y ticks
        ax.set_xticks(range(self.mapa.size))
        ax.set_yticks(range(self.mapa.size))
        ax.grid()
        
        # Crear leyenda usando artistas proxy
        legend_patches = [
            mpatches.Patch(color='white', label='Espacio vaci­o'),
            mpatches.Patch(color='black', label='Robot naive'),
            mpatches.Patch(color='green', label='Comida'),
            mpatches.Patch(color='red', label='Casa'),
            mpatches.Patch(color='purple', label='Robot Naive Pista'),
            mpatches.Patch(color='blue', label='Robot sabio')
        ]
        
        # Anyadir la leyenda fuera de la grafica
        ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        
        # Ti­tulo
        ax.set_title("t = " + str(round(self.env.now, 4)))
        
        # Guardamos la grafica en la lista de graficas
        self.mapa.lista_graficas.append(fig)
        
        plt.close(fig)
    """
    def run(self):
        try:
            # Vamos moviendo el robot por el espacio; se mueve segun una distribucion de tiempo y espacio aleatorias
                # Se puede mover arriba/abajo/izquierda/derecha
                # Se movera cada x segundos, segun una distribucion uniforme (0.5, 1.5)
                
            while self.sabio == False:
                # Como se mueve cada x segundos segun U~[1, 2]
                yield self.env.timeout(nprandom.uniform(1, 2))
                # Ahora decide a donde moverse
                with self.semaforo_cambio.request() as req:
                    yield req # Aqui­ hasta que no obtenemos el recurso nos quedamos quietos
                    
                    # Ya hemos obtenido el recurso, intentamos movernos
                    
                    # Tenemos que tener en cuenta si las probabilidades son las actuales o no por la pista
                    
                    # Si tenemos pista activa, el numero sera mayor a -1 y menor a self.duracion_pista == len(prob_maximas)
                    if self.iteracion_pista>-1 and self.iteracion_pista <self.duracion_pista:
                        
                        # Tenemos que calcular las nuevas probabilidades
                        self.calcular_probabilidades()
                        # print(self.direcciones_naive)
                        
                        # Tendremos ya la lista actualizada
                        # Actualizo su iteracion_pista (para recalcular despues las probabilidades)
                        self.iteracion_pista = self.iteracion_pista + 1
                    
                    # Si justamente es igual, es porque se ha agotado la pista
                    elif self.iteracion_pista == self.duracion_pista: 
                        
                        # Ponemos todo como al principio
                        self.direcciones_naive = [[np.array([-1, 0]), 0.25], [np.array([1, 0]), 0.25], [np.array([0, -1]), 0.25], [np.array([0, 1]), 0.25]]
                        self.iteracion_pista = -1
                        self.direccion_pista = None
                        self.mapa.robots_pista[self.name] = False
                    
                    # Si no ha entrado en ninguno de los casos anteriores es porque es -1, simplemente removemos
                    
                    # Revuelvo la lista teniendo en cuenta las probabilidades
                    self.direcciones_naive = self.revolver_ponderado(self.direcciones_naive)
                    
                    pos_direcciones = 0
                    
                    # Creo un booleano que me servira para hacer el procedimiento de moverse o no 
                    puede_moverse = False
                    
                    # Intento con cada direccion (en orden segun lista mezclada)
                    while True:
                        
                        # Cojo la direccion
                        direccion_prov = self.direcciones_naive[pos_direcciones][0]
                        
                        # Calculo a donde me movere
                        pos_hor = self.posicion[0] + direccion_prov[0]
                        pos_ver = self.posicion[1] + direccion_prov[1]
                        
                        # Primer caso con problemas: me salgo del mapa:
                            
                        if pos_hor == self.mapa.size or pos_hor < 0 or pos_ver == self.mapa.size or pos_ver < 0:
                            
                            #print('\n%s wanted to move from pos %s to pos %s but wall' % (self.name, self.posicion, [pos_hor, pos_ver]))
                            # Direccion invÃ¡lida, intento con la siguiente
                            pos_direcciones = pos_direcciones + 1
                        
                        # Segundo caso con problemas: Si no es el anterior, es que a donde me quiera mover hay un robot
                        # Si es la casa, tengo que justo querer moverme ahi­ y que haya un robot
                        #elif self.mapa.matriz[pos_hor, pos_ver] == 1 or (sum(1 for valor in self.mapa.posiciones.values() if valor == self.mapa.posiciones["Casa"]) == 2 and self.mapa.posiciones["Casa"] == [pos_hor, pos_ver]) or (sum(1 for valor in self.mapa.posiciones.values() if valor == self.mapa.posiciones["Comida"]) == 2 and self.mapa.posiciones["Comida"] == [pos_hor, pos_ver]):
                        elif (
                            self.mapa.matriz[pos_hor, pos_ver] == 1 or
                            (
                                sum(1 for valor in self.mapa.posiciones.values() if np.array_equal(valor, self.mapa.posiciones["Casa"])) > 1 and 
                                self.mapa.matriz[pos_hor, pos_ver] == 3
                            ) or 
                            (
                                sum(1 for valor in self.mapa.posiciones.values() if np.array_equal(valor, self.mapa.posiciones["Comida"])) > 1 and 
                                self.mapa.matriz[pos_hor, pos_ver] == 2
                            )
                        ):
                            # Miro cual es el robot que estaba ahi
                            #compi = next((clave for clave, valor in self.mapa.posiciones.items() if valor == [pos_hor, pos_ver]), None)
                            compi = next(
                                (clave for clave, valor in self.mapa.posiciones.items() if np.array_equal(valor, [pos_hor, pos_ver]) and clave not in ["Casa", "Comida"]),
                                None)
                            #print(compi)
                            #print('\n%s wanted to move from pos %s to pos %s but %s' % (self.name, self.posicion, [pos_hor, pos_ver], compi))
                            
                            # Le pregunto si sabe donde esta la comida
                            # Si el compi si­ que sabe
                            if self.mapa.robots_sabios[compi] == True:
                                # Le da la pista
                                self.direccion_pista = self.calcular_direccion_comida(self.mapa.posiciones["Comida"]-self.posicion)
                                #print(compi, " le ha dicho a ", self.name, " una direccion a seguir: ", self.direccion_pista)
                                self.iteracion_pista = 0
                                self.mapa.robots_pista[self.name] = True
                                #self.graficar()
                                break
                            
                            # Direccion invalida, intento con la siguiente
                            pos_direcciones = pos_direcciones + 1
                        
                        # Caso bueno: Puede que encuentre la comida (y que no este ocupada por otro robot)
                        elif self.mapa.matriz[pos_hor, pos_ver] == 2 and sum(1 for valor in self.mapa.posiciones.values() if np.array_equal(valor, self.mapa.posiciones["Comida"])) == 1:
                            #print(self.name, " ha encontrado la comida de casualidad. Quiere moverse a ", [pos_hor, pos_ver])
                            # El propio robot se vuelve sabio
                            self.mapa.robots_sabios[self.name] = True
                            self.mapa.comidas_llevadas[self.name] = 0
                            self.sabio = True
                            self.mapa.robots_pista[self.name] = False
                            self.iteracion_pista = -1
                            self.direccion_pista = None
                            
                            
                            if len(self.mapa.comidas_llevadas) == self.num_robots:
                                # Creo la gráfica
                                #self.graficar()
                                self.mapa.parar_simulacion = True
                                raise FinalizarSimulacion()
                                
                            # Empieza a ejecutar la segunda parte del run (llevar comida a casa e ir a buscarla a la fuente)
                            puede_moverse = True
                            break
                        
                        
                        # En otro caso, podra moverse o a una casilla normal vaci­a o pasar por casa vacÃ­a
                        else:
                            puede_moverse = True
                            break
                        
                        # Si no se puede mover en ninguna direccion, que salga del bucle pero que no se mueva
                        if pos_direcciones + 1 == len(self.direcciones_naive):
                            #print(f"{self.name} no puede moverse. Salta el turno.")
                            break
                    
                    
                    # Para que no se mueva donde estaba el companyero sabio
                    if puede_moverse: 
                            # Se mueve (pero si es la casa/comida no anoto el 1)
                        if self.mapa.matriz[pos_hor, pos_ver] != 3 and self.mapa.matriz[pos_hor, pos_ver] != 2:
                            self.mapa.matriz[pos_hor, pos_ver] = 1
                        # Donde estaba es un cero, si no es la casa/comida
                        if self.mapa.matriz[self.posicion[0], self.posicion[1]] != 3 and self.mapa.matriz[self.posicion[0], self.posicion[1]] != 2:
                            self.mapa.matriz[self.posicion[0], self.posicion[1]] = 0
                        
                        #print('\n%s moving from pos %s to pos %s at t= %s' % (self.name, self.posicion, [pos_hor, pos_ver], self.env.now))
                        
                        # Actualizo su posicion 
                        self.posicion = [pos_hor, pos_ver]
                        # Actualizo diccionario
                        self.mapa.posiciones[self.name] = [pos_hor, pos_ver]
                        # Creo la grafica y la anyado a la lista
                        #self.graficar()
                       
            # Si llega aqui­, es porque es sabio
            while True:
                
                # Comprueba si tiene que terminar antes de moverse
                if len(self.mapa.comidas_llevadas) == self.num_robots:
                    break
                
                # Como se mueve cada x segundos segun U~[1, 2]
                yield self.env.timeout(nprandom.uniform(1, 2))
                # Ahora decide a donde moverse
                with self.semaforo_cambio.request() as req:
                    yield req # Aqui­ hasta que no obtenemos el recurso nos quedamos quietos
                    
                    # Ya hemos obtenido el recurso, intentamos movernos
                    
                    # Antes de hacer nada, si tenemos que parar, salimos del segundo while y se acaba
                    if self.mapa.parar_simulacion == True:
                        break
                    
                    # Primero, tenemos que saber si estamos en casa/comida, porque en ese caso hay que cambiar el objetivo
                    # Si esta en casa, hay que indicarle que va a por comida
                    if np.array_equal(self.posicion, self.mapa.posiciones["Casa"]): 
                        self.lleva_comida = False
                        
                    # Si esta en comida, hay que indicarle que va a casa
                    if np.array_equal(self.posicion, self.mapa.posiciones["Comida"]):
                        self.lleva_comida = True 
                    
                    # Dependiendo de si el robot va de casa a la comida o al reves, hacemos una cosa u otra
                    # Si va a la comida
                    if self.lleva_comida == False:
                        # Calculamos cuantas casillas (y en que direccion) tiene que ir: lo llamamos ruta
                        ruta = self.mapa.comida - self.posicion
                    
                    # Si lleva la comida a casa
                    else:
                        # Calculamos cuantas casillas (y en que direccion) tiene que ir: lo llamamos ruta
                        ruta = self.mapa.casa - self.posicion
                    
                    
                    # De esta forma: 
                        # Si ruta[0] es negativo, tiene que ir ruta[0] casillas hacia arriba 
                        # Si ruta[0] es positivo, tiene que ir ruta[0] casillas hacia abajo
                        # Si ruta[0] = 0, esta en la fila
                        # Si ruta[1] es negativo, tiene que ir ruta[0] casillas hacia la izquierda
                        # Si ruta[1] es positivo, tiene que ir ruta[0] casillas hacia la derecha
                        # Si ruta[1] = 0, esta en la columna
                    
                    # Nos guardamos los posibles movimientos
                    posibles_movimientos = []
                    if ruta[0]<0:
                        posibles_movimientos.append(np.array([-1, 0]))
                    elif ruta[0]>0:
                        posibles_movimientos.append(np.array([1, 0]))
    
                    if ruta[1]<0: 
                        posibles_movimientos.append(np.array([0, -1]))
                    elif ruta[1]>0:
                        posibles_movimientos.append(np.array([0, 1]))
                    
                    # Aqui­ surgen tres posibilidades: 
                    # Si no esta ni en la fila ni en la columna del destino: 
                    if len(posibles_movimientos)==2:
                        # Hacemos un sorteo para ver a donde va
                        umbral = np.abs(ruta[0])/(np.sum(np.abs(ruta)))
                        u = nprandom.uniform()
                        if u<= umbral:
                            movimientos_finales = [posibles_movimientos[0], posibles_movimientos[1]]
                        else:
                            movimientos_finales = [posibles_movimientos[1], posibles_movimientos[0]]
                        
                    
                    # Si solo tiene una posible direccion porque ya esta en la fila/columna del destino, pero no en la propia casilla
                    elif len(posibles_movimientos)==1: 
                        movimientos_finales = posibles_movimientos
                    
                    # Por si acaso, para intentar que el robot no se quede quieto, anyado los movimientos que le falten (incluso si se aleja de donde tiene que ir)
                    # faltantes = [sublista for sublista in self.direcciones if sublista not in movimientos_finales]
                    faltantes = [sublista for sublista in self.direcciones_sabio if not any(np.array_equal(sublista, mov) for mov in movimientos_finales)]
                    nprandom.shuffle(faltantes)
                    movimientos_finales = movimientos_finales + faltantes
                    
                    #print("Robot: ",self.name, ", movimientos finales: ",movimientos_finales)
                    
                    pos_direcciones = 0
                    
                    # Creo un booleano que me servira para hacer el procedimiento de moverse o no 
                    puede_moverse = False
                    
                    # Intento con cada direccion (en orden segun lista mezclada)
                    while True:
                        
                        """
                        comprobar que pos_direcciones<len(movimientos_finales) SIEMPRE Y EN EL OTRO CAMBIARLO
                        """
                        
                        if pos_direcciones==len(movimientos_finales):
                            #print('\n%s can not move at t= %s' % (self.name, self.env.now))
                            break
                        
                        else:
                        
                            # Cojo la direccion
                            #print(pos_direcciones,len(movimientos_finales))
                            direccion_prov = movimientos_finales[pos_direcciones]
                            
                            # Calculo a donde me moveri­a
                            pos_hor = self.posicion[0] + direccion_prov[0]
                            pos_ver = self.posicion[1] + direccion_prov[1]
                        
                            # Ya sabe el robot a donde tiene que moverse. Ahora tiene que hacerlo. Surgen 3 posibilidades
                            # El robot se mueve a una casilla porque esta vacÃ­a (como sabe por donde va, no se va a dar de bruces con la pared) que no es casa/comida.
                            
                            # Primero, si se fuera del mapa
                            if pos_hor == self.mapa.size or pos_hor < 0 or pos_ver == self.mapa.size or pos_ver < 0: 
                                
                                pos_direcciones = pos_direcciones + 1
                                
                            elif sum(1 for valor in self.mapa.posiciones.values() if np.array_equal(valor, [pos_hor, pos_ver])) == 0:
                                
                                # Se podra mover
                                puede_moverse = True
                                break
                            
                            # Si es justo la casa/comida, tengo que vigilar que pueda moverme ahi­ (que en el diccionario de posiciones de mapa no haya dos claves con la pos de casa/comida)
                            # Si justo quiero llegar a casa, y solo hay un elemento en el diccionario de posiciones que tenga esa posicion (la propia casa)
                            elif sum(1 for valor in self.mapa.posiciones.values() if np.array_equal(valor, self.mapa.posiciones["Casa"])) == 1 and np.array_equal(self.mapa.posiciones["Casa"], np.array([pos_hor, pos_ver])):
                                
                                # Se podra mover
                                puede_moverse = True
                                break
                            
                            elif sum(1 for valor in self.mapa.posiciones.values() if np.array_equal(valor, self.mapa.posiciones["Comida"])) == 1 and np.array_equal(self.mapa.posiciones["Comida"], np.array([pos_hor, pos_ver])):
                                
                                # Se podra mover
                                puede_moverse = True
                                break
                                
                            else:
                                pos_direcciones = pos_direcciones + 1
                                
                    
                    if puede_moverse:
                        
                        
                        # print(self.lleva_comida, [pos_hor, pos_ver], self.mapa.comida, type(self.mapa.comida), self.mapa.matriz[pos_hor, pos_ver])
                        # print(self.mapa.matriz)
                        
                        # Comprueba si tiene que terminar antes de moverse
                        if len(self.mapa.comidas_llevadas) == self.num_robots:
                            break
                        
                        # Si estoy aqui­, se movera, ahora tengo que ver si es una casilla normal o el destino
                        # Si es normal                        
                        if self.mapa.matriz[pos_hor, pos_ver] == 0:
                            
                            # Se mueve
                            self.mapa.matriz[pos_hor, pos_ver] = 1
                            # Donde estaba es un cero (si era un uno, si no, lo dejo porque es la casa o la fuente)
                            if self.mapa.matriz[self.posicion[0], self.posicion[1]] == 1:
                                self.mapa.matriz[self.posicion[0], self.posicion[1]] = 0
                            
                            #print('\n%s moving from pos %s to pos %s at t= %s' % (self.name, self.posicion, [pos_hor, pos_ver], self.env.now))
                            
                            # Actualizo su posicion 
                            self.posicion = [pos_hor, pos_ver]
                            # Actualizo diccionario
                            self.mapa.posiciones[self.name] = [pos_hor, pos_ver]
                            
                        
                        
                        # Si esta yendo a buscar comida y es la fuente
                        elif self.lleva_comida == False and self.mapa.matriz[pos_hor, pos_ver] == 2:
                            # Donde estaba es un cero (si era un uno, si no, lo dejo porque es la casa o la fuente)
                            if self.mapa.matriz[self.posicion[0], self.posicion[1]] == 1:
                                self.mapa.matriz[self.posicion[0], self.posicion[1]] = 0
                           
                            # Actualizo su posicion 
                            self.posicion = [pos_hor, pos_ver]
                            # Actualizo diccionario
                            self.mapa.posiciones[self.name] = [pos_hor, pos_ver]
                            # Le indicamos que tiene que volver a casa
                            self.lleva_comida = True
                            #print('\n%s arrived to food loc at t= %s, lleva comida a casa: %s' % (self.name, self.env.now, self.lleva_comida))
    
                        
                        # Si esta yendo a dejar la comida en casa y es la casa
                        elif self.lleva_comida == True and self.mapa.matriz[pos_hor, pos_ver] == 3:
                        
                            # Donde estaba es un cero (si era un uno, si no, lo dejo porque es la casa o la fuente)
                            if self.mapa.matriz[self.posicion[0], self.posicion[1]] == 1:
                                self.mapa.matriz[self.posicion[0], self.posicion[1]] = 0
                            
                            # Actualizo su posicion 
                            self.posicion = [pos_hor, pos_ver]
                            # Actualizo diccionario
                            self.mapa.posiciones[self.name] = [pos_hor, pos_ver]
                            # Le indicamos que tiene que ir a buscar comida
                            self.lleva_comida = False
                            #Actualizamos comidas
                            self.mapa.comidas_llevadas[self.name] = self.mapa.comidas_llevadas[self.name] + 1
                            #print('\n%s arrived home at t= %s, lleva comida: %s' % (self.name, self.env.now, self.lleva_comida))
                            #print(self.mapa.comidas_llevadas)
                        
                        # En cualquiera de los casos, creo la grafica y la aÃ±ado a la lista
                        #self.graficar()
            
        except FinalizarSimulacion:
            gato = 3
            #print("SE ACABA LA SIMULACION PORQUE TODOS LOS ROBOTS SON SABIOS.")

# Para un tamaño de mapa determinado, para una distancia mínima entre la casa y la comida y para un número de robots
# Probamos con diferentes combinaciones de "prob inicial pista" y de "duracion pista" 

# OJO CON SIZE Y DIST: NO VAYA A SER QUE NO SE PUEDA PONER LA COMIDA POR NO HABER NINGUNA CASILLA A MAYOR DISTANCIA QUE DIST

size = 6
dist = 3
num_robots = 5
num_experimentos = 200
# IGUAL NO PONER 1? SI ESO 0.999999
lista_prob_inicial_pista = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.999]
lista_duracion_pista = [1, 2, 3, 4]
lista_resultados = []

for a in lista_prob_inicial_pista: 
    for b in lista_duracion_pista:
        print(a,b)
        tiempo_total = 0
        
        for experimento in range(num_experimentos):
            #print(experimento)
            
            #env = simpy.rt.RealtimeEnvironment(factor=0.5)
            env = simpy.Environment()
            semaforo_cambio = simpy.Resource(env, capacity=1)
            
            mapa = Mapa(size, dist)

            prob_inicial_pista = a
            duracion_pista = b
            lista_robots = [] 
            for i in range(num_robots):
                lista_robots.append(Robot(env, 'Robot %d' % i, mapa, semaforo_cambio, num_robots, prob_inicial_pista, duracion_pista))

            env.run()
            tiempo_total = tiempo_total + env.now
        
        lista_resultados.append([a,b,tiempo_total/num_experimentos])
        #print("Resultados de prob_inicial_pista = ",a,", duracion_pista: ",b)
        #print("Tiempo medio = ",tiempo_total/num_experimentos,", comidas llevadas media = ",comidas_llevadas/num_experimentos)

lista_resultados = sorted(lista_resultados, key=lambda x: x[2])
for lista in lista_resultados:
    print(lista)


#from mpl_toolkits.mplot3d import Axes3D

# Separar los elementos en listas individuales para cada eje
x = [item[0] for item in lista_resultados]
y = [item[1] for item in lista_resultados]
z = [item[2] for item in lista_resultados]


# SI QUIERO HACER UNA SUPERFICIE
# Crear una figura 3D
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111, projection='3d')

xi = np.linspace(min(x), max(x), len(x))
yi = np.linspace(min(y), max(y), len(y))
xi, yi = np.meshgrid(xi, yi)

# Interpolar los valores de Z para crear una superficie
from scipy.interpolate import griddata
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Crear la superficie
fig_sup = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi)])

# Personalizar el diseño
fig_sup.update_layout(
    title="Gráfica de Superficie 3D",
    scene=dict(
        xaxis_title="Eje X",
        yaxis_title="Eje Y",
        zaxis_title="Eje Z"
    )
)

# Guardar la gráfica en un archivo HTML temporal
file_path_sup = "temp_plot_superficie.html"
fig_sup.write_html(file_path_sup)

# Abrir la gráfica en una pestaña adicional del navegador
webbrowser.open(f"file://{os.path.abspath(file_path_sup)}")

# SI QUIERO HACER CON PUNTOS

fig_scat = go.Figure(data=[go.Scatter3d(
    x=x, 
    y=y, 
    z=z,
    mode='markers',  # 'markers' para puntos, 'lines' para líneas
    marker=dict(
        size=8,
        color=z,  # Color basado en los valores de Z
        colorscale='Viridis',  # Escala de colores
        opacity=0.8
    )
)])


# Personalizar el diseño
fig_scat.update_layout(
    title="Gráfica Scatter 3D",
    scene=dict(
        xaxis_title="Eje X",
        yaxis_title="Eje Y",
        zaxis_title="Eje Z"
    )
)

# Guardar la gráfica en un archivo HTML temporal
file_path_scat = "temp_plot_scat.html"
fig_scat.write_html(file_path_scat)

# Abrir la gráfica en una pestaña adicional del navegador
webbrowser.open(f"file://{os.path.abspath(file_path_scat)}") 

"""
# Graficar la superficie con un mapa de colores
surf = ax.plot_surface(xi, yi, zi, cmap='coolwarm', edgecolor='none')

# Añadir la barra de color
fig.colorbar(surf, ax=ax, label='Altura (Z)')

# Etiquetas de los ejes
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')

# Mostrar el gráfico
plt.show()
"""


"""

# Crear el GIF
output_path = "./simulacion.gif"

images = []

for fig in mapa.lista_graficas:
    # Ajustar margenes automaticamente
    fig.tight_layout()
    # Guardar la figura en un objeto BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)  # Regresar al inicio del buffer
    # Convertir la imagen a formato PIL antes de cerrar el buffer
    image = Image.open(buf).copy()  # Copiar la imagen para evitar problemas al cerrar el buffer
    images.append(image)
    buf.close()  # Cerrar el buffer despues de la copia

images[0].save(
    output_path, 
    save_all=True, 
    append_images=images[1:], 
    duration=250,  # Duracion de cada frame en milisegundos
    loop=0  # 0 para bucle infinito
)
"""


