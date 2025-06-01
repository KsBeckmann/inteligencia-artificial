import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import copy
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Item:
    """Representa um item da mochila"""
    id: int
    peso: int
    valor: int
    ratio: float = 0.0
    
    def __post_init__(self):
        self.ratio = self.valor / self.peso if self.peso > 0 else 0

class KnapsackProblem:
    """Representa uma instância do problema da mochila"""
    
    def __init__(self, items: List[Item], capacidade: int):
        self.items = items
        self.capacidade = capacidade
        self.n_items = len(items)
    
    @classmethod
    def from_csv(cls, filepath: str):
        """Carrega problema da mochila de um arquivo CSV"""
        df = pd.read_csv(filepath)
        
        # Remove a linha da capacidade do DataFrame principal
        capacity_row = df[df['Item'] == 'Capacidade da Mochila']
        capacidade = int(capacity_row['Peso'].iloc[0])
        
        # Remove a linha da capacidade
        df = df[df['Item'] != 'Capacidade da Mochila']
        
        items = []
        for _, row in df.iterrows():
            item = Item(
                id=int(row['Item']),
                peso=int(row['Peso']),
                valor=int(row['Valor'])
            )
            items.append(item)
        
        return cls(items, capacidade)

class GeneticAlgorithm:
    """Implementação do Algoritmo Genético para o problema da mochila"""
    
    def __init__(self, problem: KnapsackProblem, config: Dict[str, Any]):
        self.problem = problem
        self.config = config
        self.population = []
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []
        self.generation = 0
        
    def initialize_population(self, method='random'):
        """Inicializa a população"""
        self.population = []
        pop_size = self.config['population_size']
        
        if method == 'random':
            for _ in range(pop_size):
                individual = [random.randint(0, 1) for _ in range(self.problem.n_items)]
                self.population.append(individual)
        
        elif method == 'heuristic':
            # Parte da população baseada em heurística (ratio valor/peso)
            sorted_items = sorted(enumerate(self.problem.items), 
                                key=lambda x: x[1].ratio, reverse=True)
            
            heuristic_size = pop_size // 2
            
            # Indivíduos heurísticos
            for _ in range(heuristic_size):
                individual = [0] * self.problem.n_items
                current_weight = 0
                
                for idx, item in sorted_items:
                    if current_weight + item.peso <= self.problem.capacidade:
                        if random.random() < 0.8:  # 80% chance de incluir item bom
                            individual[idx] = 1
                            current_weight += item.peso
                
                self.population.append(individual)
            
            # Resto da população aleatória
            for _ in range(pop_size - heuristic_size):
                individual = [random.randint(0, 1) for _ in range(self.problem.n_items)]
                self.population.append(individual)
    
    def fitness(self, individual):
        """Calcula o fitness de um indivíduo"""
        total_weight = sum(individual[i] * self.problem.items[i].peso 
                          for i in range(self.problem.n_items))
        total_value = sum(individual[i] * self.problem.items[i].valor 
                         for i in range(self.problem.n_items))
        
        # Penaliza soluções que excedem a capacidade
        if total_weight > self.problem.capacidade:
            penalty = (total_weight - self.problem.capacidade) * 10
            return max(0, total_value - penalty)
        
        return total_value
    
    def tournament_selection(self, tournament_size=3):
        """Seleção por torneio"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=self.fitness)
    
    def crossover_one_point(self, parent1, parent2):
        """Crossover de um ponto"""
        if random.random() > self.config['crossover_rate']:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def crossover_two_point(self, parent1, parent2):
        """Crossover de dois pontos"""
        if random.random() > self.config['crossover_rate']:
            return parent1.copy(), parent2.copy()
        
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2
    
    def crossover_uniform(self, parent1, parent2):
        """Crossover uniforme"""
        if random.random() > self.config['crossover_rate']:
            return parent1.copy(), parent2.copy()
        
        child1 = []
        child2 = []
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        return child1, child2
    
    def mutate(self, individual):
        """Aplica mutação em um indivíduo"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.config['mutation_rate']:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated
    
    def run(self):
        """Executa o algoritmo genético"""
        # Inicialização
        self.initialize_population(self.config['initialization'])
        
        # Avalia população inicial
        fitness_values = [self.fitness(ind) for ind in self.population]
        best_idx = np.argmax(fitness_values)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        self.fitness_history.append(self.best_fitness)
        
        stagnation_count = 0
        
        for generation in range(self.config['max_generations']):
            self.generation = generation
            new_population = []
            
            # Elitismo - mantém os melhores indivíduos
            elite_size = int(self.config['population_size'] * 0.1)
            fitness_values = [self.fitness(ind) for ind in self.population]
            elite_indices = np.argsort(fitness_values)[-elite_size:]
            
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
            
            # Gera nova população
            while len(new_population) < self.config['population_size']:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Aplica crossover
                if self.config['crossover_type'] == 'one_point':
                    child1, child2 = self.crossover_one_point(parent1, parent2)
                elif self.config['crossover_type'] == 'two_point':
                    child1, child2 = self.crossover_two_point(parent1, parent2)
                else:  # uniform
                    child1, child2 = self.crossover_uniform(parent1, parent2)
                
                # Aplica mutação
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Ajusta tamanho da população
            self.population = new_population[:self.config['population_size']]
            
            # Avalia nova população
            fitness_values = [self.fitness(ind) for ind in self.population]
            current_best_fitness = max(fitness_values)
            
            if current_best_fitness > self.best_fitness:
                best_idx = np.argmax(fitness_values)
                self.best_individual = self.population[best_idx].copy()
                self.best_fitness = current_best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            self.fitness_history.append(self.best_fitness)
            
            # Critério de parada por convergência
            if (self.config['stop_criterion'] == 'convergence' and 
                stagnation_count >= self.config['stagnation_limit']):
                print(f"Convergência atingida na geração {generation}")
                break
        
        return self.best_individual, self.best_fitness

def run_experiments():
    """Executa experimentos com diferentes configurações"""
    
    # Carrega problemas da pasta ed03-mochila
    problems = {}
    base_path = Path('ed03-mochila')
    
    if not base_path.exists():
        print(f"Pasta 'ed03-mochila' não encontrada!")
        print("Tentando carregar da pasta atual...")
        base_path = Path('.')
    
    for i in range(1, 11):
        try:
            filepath = base_path / f'knapsack_{i}.csv'
            problems[f'knapsack_{i}'] = KnapsackProblem.from_csv(str(filepath))
            print(f"Carregado: {filepath}")
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {filepath}")
    
    if not problems:
        print("Nenhum arquivo de problema encontrado!")
        return
    
    # Configurações para testar
    configurations = [
        {
            'name': 'Base - Um Ponto + Mutação Baixa',
            'population_size': 100,
            'max_generations': 200,
            'crossover_type': 'one_point',
            'crossover_rate': 0.8,
            'mutation_rate': 0.01,
            'initialization': 'random',
            'stop_criterion': 'generations',
            'stagnation_limit': 50
        },
        {
            'name': 'Dois Pontos + Mutação Média',
            'population_size': 100,
            'max_generations': 200,
            'crossover_type': 'two_point',
            'crossover_rate': 0.8,
            'mutation_rate': 0.05,
            'initialization': 'random',
            'stop_criterion': 'generations',
            'stagnation_limit': 50
        },
        {
            'name': 'Uniforme + Mutação Alta',
            'population_size': 100,
            'max_generations': 200,
            'crossover_type': 'uniform',
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'initialization': 'random',
            'stop_criterion': 'generations',
            'stagnation_limit': 50
        },
        {
            'name': 'Heurística + Convergência',
            'population_size': 100,
            'max_generations': 200,
            'crossover_type': 'one_point',
            'crossover_rate': 0.8,
            'mutation_rate': 0.02,
            'initialization': 'heuristic',
            'stop_criterion': 'convergence',
            'stagnation_limit': 30
        }
    ]
    
    results = {}
    
    # Executa experimentos
    for config in configurations:
        print(f"\n=== Testando configuração: {config['name']} ===")
        config_results = {}
        
        for problem_name, problem in problems.items():
            print(f"Resolvendo {problem_name}...")
            
            # Múltiplas execuções para estatísticas
            fitness_values = []
            execution_times = []
            
            for run in range(5):  # 5 execuções por configuração
                import time
                start_time = time.time()
                
                ga = GeneticAlgorithm(problem, config)
                best_solution, best_fitness = ga.run()
                
                end_time = time.time()
                
                fitness_values.append(best_fitness)
                execution_times.append(end_time - start_time)
            
            config_results[problem_name] = {
                'best_fitness': max(fitness_values),
                'avg_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'avg_time': np.mean(execution_times)
            }
            
            print(f"  Melhor fitness: {max(fitness_values)}")
            print(f"  Fitness médio: {np.mean(fitness_values):.2f}")
        
        results[config['name']] = config_results
    
    # Análise dos resultados
    print("\n" + "="*80)
    print("ANÁLISE COMPARATIVA DOS RESULTADOS")
    print("="*80)
    
    # Cria DataFrame para análise
    analysis_data = []
    for config_name, config_results in results.items():
        for problem_name, metrics in config_results.items():
            analysis_data.append({
                'Configuração': config_name,
                'Problema': problem_name,
                'Melhor Fitness': metrics['best_fitness'],
                'Fitness Médio': metrics['avg_fitness'],
                'Desvio Padrão': metrics['std_fitness'],
                'Tempo Médio (s)': metrics['avg_time']
            })
    
    df_results = pd.DataFrame(analysis_data)
    
    # Resumo por configuração
    print("\nRESUMO POR CONFIGURAÇÃO:")
    summary = df_results.groupby('Configuração').agg({
        'Melhor Fitness': ['mean', 'std'],
        'Fitness Médio': 'mean',
        'Tempo Médio (s)': 'mean'
    }).round(2)
    print(summary)
    
    # Ranking das configurações
    print("\nRANKING DAS CONFIGURAÇÕES (por fitness médio):")
    ranking = df_results.groupby('Configuração')['Fitness Médio'].mean().sort_values(ascending=False)
    for i, (config, avg_fitness) in enumerate(ranking.items(), 1):
        print(f"{i}. {config}: {avg_fitness:.2f}")
    
    # Visualização
    plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Fitness médio por configuração
    plt.subplot(2, 2, 1)
    config_means = df_results.groupby('Configuração')['Fitness Médio'].mean()
    config_means.plot(kind='bar', rot=45)
    plt.title('Fitness Médio por Configuração')
    plt.ylabel('Fitness Médio')
    plt.tight_layout()
    
    # Gráfico 2: Tempo de execução por configuração
    plt.subplot(2, 2, 2)
    time_means = df_results.groupby('Configuração')['Tempo Médio (s)'].mean()
    time_means.plot(kind='bar', rot=45, color='orange')
    plt.title('Tempo Médio de Execução')
    plt.ylabel('Tempo (segundos)')
    plt.tight_layout()
    
    # Gráfico 3: Dispersão fitness vs tempo
    plt.subplot(2, 2, 3)
    for config in df_results['Configuração'].unique():
        config_data = df_results[df_results['Configuração'] == config]
        plt.scatter(config_data['Tempo Médio (s)'], config_data['Fitness Médio'], 
                   label=config, alpha=0.7)
    plt.xlabel('Tempo Médio (s)')
    plt.ylabel('Fitness Médio')
    plt.title('Fitness vs Tempo de Execução')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gráfico 4: Boxplot de fitness por configuração
    plt.subplot(2, 2, 4)
    df_results.boxplot(column='Fitness Médio', by='Configuração', ax=plt.gca())
    plt.title('Distribuição do Fitness por Configuração')
    plt.suptitle('')  # Remove título automático do boxplot
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('genetic_algorithm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, df_results

def demonstrate_single_run():
    """Demonstra uma execução única detalhada"""
    print("DEMONSTRAÇÃO DE EXECUÇÃO ÚNICA")
    print("="*50)
    
    # Carrega o primeiro problema disponível da pasta ed03-mochila
    base_path = Path('ed03-mochila')
    
    if not base_path.exists():
        print(f"Pasta 'ed03-mochila' não encontrada! Tentando pasta atual...")
        base_path = Path('.')
    
    try:
        filepath = base_path / 'knapsack_1.csv'
        problem = KnapsackProblem.from_csv(str(filepath))
        print(f"Problema carregado de: {filepath}")
        print(f"Detalhes: {problem.n_items} itens, capacidade {problem.capacidade}")
        
        # Mostra os itens
        print("\nItens disponíveis:")
        for item in problem.items:
            print(f"Item {item.id}: Peso={item.peso}, Valor={item.valor}, Ratio={item.ratio:.3f}")
        
        # Configuração para demonstração
        config = {
            'population_size': 50,
            'max_generations': 100,
            'crossover_type': 'one_point',
            'crossover_rate': 0.8,
            'mutation_rate': 0.02,
            'initialization': 'heuristic',
            'stop_criterion': 'generations',
            'stagnation_limit': 20
        }
        
        print(f"\nConfigurações do AG:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Executa o algoritmo
        print(f"\nExecutando algoritmo genético...")
        ga = GeneticAlgorithm(problem, config)
        best_solution, best_fitness = ga.run()
        
        # Mostra resultados
        print(f"\nMelhor solução encontrada:")
        print(f"Fitness: {best_fitness}")
        
        selected_items = [i for i, selected in enumerate(best_solution) if selected]
        total_weight = sum(problem.items[i].peso for i in selected_items)
        total_value = sum(problem.items[i].valor for i in selected_items)
        
        print(f"Itens selecionados: {[problem.items[i].id for i in selected_items]}")
        print(f"Peso total: {total_weight}/{problem.capacidade}")
        print(f"Valor total: {total_value}")
        
        # Gráfico da evolução
        plt.figure(figsize=(10, 6))
        plt.plot(ga.fitness_history)
        plt.title('Evolução do Fitness ao Longo das Gerações')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)
        plt.show()
        
    except FileNotFoundError:
        print(f"Arquivo knapsack_1.csv não encontrado em {base_path}!")
        print("Certifique-se de que os arquivos CSV estão na pasta 'ed03-mochila'")
        
        # Lista arquivos disponíveis para debug
        print(f"\nArquivos encontrados em {base_path}:")
        try:
            for file in base_path.glob("*.csv"):
                print(f"  {file.name}")
        except:
            print("  Não foi possível listar os arquivos")

if __name__ == "__main__":
    print("ALGORITMO GENÉTICO PARA PROBLEMA DA MOCHILA")
    print("="*60)
    
    # Demonstração única
    demonstrate_single_run()
    
    print("\n" + "="*60)
    
    # Experimentos comparativos
    results, df_results = run_experiments()
    
    print("\nExperimentos concluídos! Verifique os gráficos gerados.")