import numpy as np
import time
import psutil
import os
import csv
from collections import deque
import heapq
import copy

class Puzzle:
    """Classe para representar o quebra-cabeça dos 8 números"""
    
    # Estado objetivo: o tabuleiro resolvido
    OBJETIVO = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    
    def __init__(self, estado):
        """Inicializa o quebra-cabeça com um estado"""
        if isinstance(estado, list):
            # Converter lista 1D para matriz 3x3
            self.estado = np.array(estado).reshape(3, 3)
        else:
            self.estado = np.array(estado)
    
    def __eq__(self, outro):
        """Verifica se dois estados são iguais"""
        return np.array_equal(self.estado, outro.estado)
    
    def __hash__(self):
        """Retorna um hash para o estado atual"""
        return hash(str(self.estado.flatten()))
    
    def __lt__(self, outro):
        """Comparação para heapq"""
        return False
    
    def copiar(self):
        """Retorna uma cópia do quebra-cabeça"""
        return Puzzle(self.estado.copy())
    
    def encontrar_espaco_vazio(self):
        """Encontra a posição do espaço vazio (0)"""
        pos = np.where(self.estado == 0)
        return pos[0][0], pos[1][0]
    
    def mover(self, direcao):
        """
        Tenta mover o espaço vazio na direção especificada
        Retorna None se o movimento for inválido
        """
        i, j = self.encontrar_espaco_vazio()
        
        # Definir o movimento para cada direção
        movimentos = {
            'cima': (-1, 0),
            'baixo': (1, 0),
            'esquerda': (0, -1),
            'direita': (0, 1)
        }
        
        di, dj = movimentos[direcao]
        novo_i, novo_j = i + di, j + dj
        
        # Verificar se o movimento é válido
        if 0 <= novo_i < 3 and 0 <= novo_j < 3:
            # Criar uma cópia do estado atual
            novo_estado = self.estado.copy()
            # Trocar o espaço vazio com a peça na direção especificada
            novo_estado[i, j], novo_estado[novo_i, novo_j] = novo_estado[novo_i, novo_j], novo_estado[i, j]
            return Puzzle(novo_estado)
        else:
            return None
    
    def gerar_sucessores(self):
        """Gera todos os possíveis estados sucessores"""
        sucessores = []
        for direcao in ['cima', 'baixo', 'esquerda', 'direita']:
            sucessor = self.mover(direcao)
            if sucessor:
                sucessores.append(sucessor)
        return sucessores
    
    def esta_resolvido(self):
        """Verifica se o quebra-cabeça está no estado objetivo"""
        return np.array_equal(self.estado, self.OBJETIVO)
    
    def distancia_manhattan(self):
        """Calcula a distância Manhattan total de todas as peças até suas posições objetivo"""
        distancia = 0
        for i in range(3):
            for j in range(3):
                valor = self.estado[i, j]
                if valor != 0:
                    # Encontrar a posição do valor no estado objetivo (linha_obj, coluna_obj)
                    pos_obj = np.where(self.OBJETIVO == valor)
                    linha_obj, coluna_obj = pos_obj[0][0], pos_obj[1][0]
                    # Calcular a distância Manhattan
                    distancia += abs(i - linha_obj) + abs(j - coluna_obj)
        return distancia
    
    def pecas_fora_do_lugar(self):
        """Calcula o número de peças fora de suas posições objetivo"""
        return np.sum(self.estado != self.OBJETIVO) - (1 if self.estado[2, 2] != 0 else 0)


class Solucao:
    """Classe para armazenar a solução e suas métricas"""
    
    def __init__(self, caminho=None, expandidos=0, fronteira_max=0, tempo=0, memoria=0):
        self.caminho = caminho if caminho else []
        self.expandidos = expandidos
        self.fronteira_max = fronteira_max
        self.tempo = tempo
        self.memoria = memoria
    
    def __str__(self):
        """Representação string da solução"""
        return f"Comprimento caminho: {len(self.caminho)}, Nós expandidos: {self.expandidos}, " \
               f"Tamanho máximo fronteira: {self.fronteira_max}, Tempo: {self.tempo:.6f}s, " \
               f"Memória: {self.memoria:.2f}MB"


class Algoritmos:
    """Classe com os algoritmos de busca"""
    
    @staticmethod
    def bfs(estado_inicial):
        """Busca em Largura (BFS)"""
        inicio = time.time()
        inicial = Puzzle(estado_inicial)
        
        if inicial.esta_resolvido():
            return Solucao([inicial], 0, 0, 0, 0)
        
        fronteira = deque([inicial])
        explorados = set()
        pai = {hash(inicial): None}
        acao = {hash(inicial): None}
        
        nos_expandidos = 0
        fronteira_max = 1
        
        while fronteira:
            fronteira_max = max(fronteira_max, len(fronteira))
            estado = fronteira.popleft()
            explorados.add(hash(estado))
            
            nos_expandidos += 1
            
            for sucessor in estado.gerar_sucessores():
                if hash(sucessor) not in explorados and sucessor not in fronteira:
                    if sucessor.esta_resolvido():
                        # Reconstruir o caminho
                        caminho = [sucessor]
                        atual = estado
                        while atual is not None:
                            caminho.append(atual)
                            atual_hash = hash(atual)
                            atual = pai.get(atual_hash)
                        
                        caminho.reverse()
                        
                        tempo = time.time() - inicio
                        memoria = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
                        
                        return Solucao(caminho, nos_expandidos, fronteira_max, tempo, memoria)
                    
                    fronteira.append(sucessor)
                    pai[hash(sucessor)] = estado
        
        # Se não encontrou solução
        tempo = time.time() - inicio
        memoria = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        
        return Solucao(None, nos_expandidos, fronteira_max, tempo, memoria)
    
    @staticmethod
    def dfs(estado_inicial, limite_profundidade=50):
        """Busca em Profundidade (DFS) limitada"""
        inicio = time.time()
        inicial = Puzzle(estado_inicial)
        
        if inicial.esta_resolvido():
            return Solucao([inicial], 0, 0, 0, 0)
        
        fronteira = [inicial]  # Usamos lista como pilha
        explorados = set()
        pai = {hash(inicial): None}
        
        nos_expandidos = 0
        fronteira_max = 1
        
        # Manter controle da profundidade
        profundidade = {hash(inicial): 0}
        
        while fronteira:
            fronteira_max = max(fronteira_max, len(fronteira))
            estado = fronteira.pop()  # Remove do final (LIFO)
            
            if hash(estado) in explorados:
                continue
                
            explorados.add(hash(estado))
            nos_expandidos += 1
            
            # Verificar limite de profundidade
            prof_atual = profundidade[hash(estado)]
            if prof_atual >= limite_profundidade:
                continue
            
            # Gerar sucessores em ordem reversa para priorizar certos movimentos
            sucessores = estado.gerar_sucessores()
            sucessores.reverse()  # Para manter a ordem correta ao usar a pilha
            
            for sucessor in sucessores:
                if hash(sucessor) not in explorados and sucessor not in fronteira:
                    if sucessor.esta_resolvido():
                        # Reconstruir o caminho
                        caminho = [sucessor]
                        atual = estado
                        while atual is not None:
                            caminho.append(atual)
                            atual_hash = hash(atual)
                            atual = pai.get(atual_hash)
                        
                        caminho.reverse()
                        
                        tempo = time.time() - inicio
                        memoria = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
                        
                        return Solucao(caminho, nos_expandidos, fronteira_max, tempo, memoria)
                    
                    fronteira.append(sucessor)
                    pai[hash(sucessor)] = estado
                    profundidade[hash(sucessor)] = prof_atual + 1
        
        # Se não encontrou solução
        tempo = time.time() - inicio
        memoria = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        
        return Solucao(None, nos_expandidos, fronteira_max, tempo, memoria)
    
    @staticmethod
    def busca_gulosa(estado_inicial):
        """Busca Gulosa usando distância Manhattan como heurística"""
        inicio = time.time()
        inicial = Puzzle(estado_inicial)
        
        if inicial.esta_resolvido():
            return Solucao([inicial], 0, 0, 0, 0)
        
        # Fila de prioridade usando distância Manhattan
        fronteira = [(inicial.distancia_manhattan(), 0, inicial)]  # (f, contador, estado)
        contador = 1  # Para garantir a estabilidade do heap
        explorados = set()
        pai = {hash(inicial): None}
        
        nos_expandidos = 0
        fronteira_max = 1
        
        while fronteira:
            fronteira_max = max(fronteira_max, len(fronteira))
            _, _, estado = heapq.heappop(fronteira)
            
            if estado.esta_resolvido():
                # Reconstruir o caminho
                caminho = [estado]
                atual = pai[hash(estado)]
                while atual is not None:
                    caminho.append(atual)
                    atual_hash = hash(atual)
                    atual = pai.get(atual_hash)
                
                caminho.reverse()
                
                tempo = time.time() - inicio
                memoria = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
                
                return Solucao(caminho, nos_expandidos, fronteira_max, tempo, memoria)
            
            if hash(estado) in explorados:
                continue
                
            explorados.add(hash(estado))
            nos_expandidos += 1
            
            for sucessor in estado.gerar_sucessores():
                if hash(sucessor) not in explorados:
                    pai[hash(sucessor)] = estado
                    h = sucessor.distancia_manhattan()
                    heapq.heappush(fronteira, (h, contador, sucessor))
                    contador += 1
        
        # Se não encontrou solução
        tempo = time.time() - inicio
        memoria = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        
        return Solucao(None, nos_expandidos, fronteira_max, tempo, memoria)
    
    @staticmethod
    def a_star(estado_inicial):
        """Algoritmo A* usando distância Manhattan como heurística"""
        inicio = time.time()
        inicial = Puzzle(estado_inicial)
        
        if inicial.esta_resolvido():
            return Solucao([inicial], 0, 0, 0, 0)
        
        # Fila de prioridade para A*: f(n) = g(n) + h(n)
        fronteira = [(inicial.distancia_manhattan(), 0, inicial)]  # (f, contador, estado)
        contador = 1  # Para garantir a estabilidade do heap
        explorados = set()
        pai = {hash(inicial): None}
        
        # Custo do caminho desde o início
        g_score = {hash(inicial): 0}
        
        nos_expandidos = 0
        fronteira_max = 1
        
        while fronteira:
            fronteira_max = max(fronteira_max, len(fronteira))
            _, _, estado = heapq.heappop(fronteira)
            
            if estado.esta_resolvido():
                # Reconstruir o caminho
                caminho = [estado]
                atual = pai[hash(estado)]
                while atual is not None:
                    caminho.append(atual)
                    atual_hash = hash(atual)
                    atual = pai.get(atual_hash)
                
                caminho.reverse()
                
                tempo = time.time() - inicio
                memoria = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
                
                return Solucao(caminho, nos_expandidos, fronteira_max, tempo, memoria)
            
            if hash(estado) in explorados:
                continue
                
            explorados.add(hash(estado))
            nos_expandidos += 1
            
            for sucessor in estado.gerar_sucessores():
                # g(n) é custo até agora + 1 passo
                g_tentativo = g_score[hash(estado)] + 1
                
                sucessor_hash = hash(sucessor)
                if sucessor_hash not in g_score or g_tentativo < g_score[sucessor_hash]:
                    # Atualizar o caminho
                    pai[sucessor_hash] = estado
                    g_score[sucessor_hash] = g_tentativo
                    
                    # f(n) = g(n) + h(n)
                    f = g_tentativo + sucessor.distancia_manhattan()
                    heapq.heappush(fronteira, (f, contador, sucessor))
                    contador += 1
        
        # Se não encontrou solução
        tempo = time.time() - inicio
        memoria = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        
        return Solucao(None, nos_expandidos, fronteira_max, tempo, memoria)


def formatar_tabuleiro(estado):
    """Formata o tabuleiro para visualização"""
    resultado = ""
    for i in range(3):
        linha = ""
        for j in range(3):
            valor = estado[i, j]
            if valor == 0:
                linha += " _ "
            else:
                linha += f" {valor} "
        resultado += linha + "\n"
    return resultado


def imprimir_caminho(solucao):
    """Imprime o caminho da solução"""
    if not solucao.caminho:
        print("Não foi encontrada solução.")
        return
    
    print(f"Número de passos: {len(solucao.caminho) - 1}")
    for i, estado in enumerate(solucao.caminho):
        print(f"\nPasso {i}:")
        print(formatar_tabuleiro(estado.estado))


def carregar_instancias(arquivo):
    """Carrega instâncias do problema de um arquivo CSV"""
    instancias = []
    with open(arquivo, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Pular cabeçalho
        
        for row in reader:
            instancia = [int(val) for val in row]
            instancias.append(instancia)
    
    return instancias


def executar_experimentos(instancias):
    """Executa experimentos com diferentes algoritmos para todas as instâncias"""
    resultados = []
    
    algoritmos = [
        ("BFS", Algoritmos.bfs),
        ("DFS", Algoritmos.dfs),
        ("Busca Gulosa", Algoritmos.busca_gulosa),
        ("A*", Algoritmos.a_star)
    ]
    
    for i, instancia in enumerate(instancias):
        print(f"\n--- Instância {i+1} ---")
        print("Estado inicial:")
        print(formatar_tabuleiro(np.array(instancia).reshape(3, 3)))
        
        instancia_resultados = {"instancia": i+1}
        
        for nome, algoritmo in algoritmos:
            print(f"\nExecutando {nome}...")
            solucao = algoritmo(instancia)
            
            print(f"Resultado {nome}:")
            print(solucao)
            
            if solucao.caminho:
                print(f"Solução encontrada em {len(solucao.caminho)-1} passos")
            else:
                print("Não foi encontrada solução.")
            
            instancia_resultados[nome] = {
                "expandidos": solucao.expandidos,
                "fronteira_max": solucao.fronteira_max,
                "tempo": solucao.tempo,
                "memoria": solucao.memoria,
                "passos": len(solucao.caminho)-1 if solucao.caminho else None
            }
        
        resultados.append(instancia_resultados)
    
    return resultados


def analisar_resultados(resultados):
    """Realiza análise comparativa dos resultados"""
    algoritmos = ["BFS", "DFS", "Busca Gulosa", "A*"]
    metricas = ["expandidos", "fronteira_max", "tempo", "memoria", "passos"]
    
    # Médias por algoritmo
    print("\n--- Médias por Algoritmo ---")
    for algoritmo in algoritmos:
        print(f"\n{algoritmo}:")
        for metrica in metricas:
            valores = [r[algoritmo][metrica] for r in resultados if r[algoritmo][metrica] is not None]
            if valores:
                media = sum(valores) / len(valores)
                print(f"Média {metrica}: {media:.4f}")
    
    # Melhor algoritmo por métrica
    print("\n--- Melhor Algoritmo por Métrica ---")
    for metrica in metricas:
        if metrica == "passos":
            # Para passos, menor é melhor, mas None deve ser ignorado
            melhor_algo = None
            melhor_valor = float('inf')
            
            for algoritmo in algoritmos:
                valores = [r[algoritmo][metrica] for r in resultados if r[algoritmo][metrica] is not None]
                if valores:
                    media = sum(valores) / len(valores)
                    if media < melhor_valor:
                        melhor_valor = media
                        melhor_algo = algoritmo
            
            if melhor_algo:
                print(f"Melhor algoritmo para {metrica}: {melhor_algo} (média: {melhor_valor:.4f})")
        
        elif metrica in ["expandidos", "fronteira_max", "tempo", "memoria"]:
            # Para estas métricas, menor é melhor
            melhor_algo = min(algoritmos, key=lambda a: sum(r[a][metrica] for r in resultados) / len(resultados))
            media = sum(r[melhor_algo][metrica] for r in resultados) / len(resultados)
            print(f"Melhor algoritmo para {metrica}: {melhor_algo} (média: {media:.4f})")


def main():
    """Função principal"""
    # Carregar instâncias do arquivo CSV
    print("Carregando instâncias do arquivo...")
    instancias = carregar_instancias("ed02-puzzle8.csv")
    print(f"Carregadas {len(instancias)} instâncias.")
    
    # Executar experimentos
    resultados = executar_experimentos(instancias)
    
    # Analisar resultados
    print("\n\n=== ANÁLISE DE RESULTADOS ===")
    analisar_resultados(resultados)


if __name__ == "__main__":
    main()