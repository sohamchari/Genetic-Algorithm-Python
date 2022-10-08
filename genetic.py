"""
Basic Genetic Algorithm

function GENETIC_ALGORITHM(population, fitness) -> Individual
    repeat
        weights <- WEIGHTED-BY(population, fitness)
        population2 <- empty list
        for i = 1 to size(population) do
            parent1, parent2 <- WEIGHTED-RANDOM-CHOICES(population, weights, 2)
            child <- REPRODUCE(parent1, parent2)
            if (small random probability) then child <- MUTATE(child)
            add child to population2
        population <- population2
    until some individual is fit enough, or enough time has elapsed
    return the best individual in population according to fitness

function REPRODUCE(parent1, parent2) -> Individual
    n <- LENGTH(parent1)
    c <- random number from 1 to n
    return APPEND(SUBSTRING(parent1, 1, c), SUBSTRING(parent2, c + 1, n))

"""

from random import choices, randint, randrange, random
from time import time
import matplotlib.pyplot as plt
from Graph_Creator import Graph_Creator


def generateGenome(length):
    return choices(['r', 'g', 'b'], k=length)


def generatePopulation(populationSize, genomeLength):
    return [generateGenome(genomeLength) for _ in range(populationSize)]


# def crossover(parent1, parent2):
#     if len(parent1) != len(parent2):
#         raise ValueError("Both parents must be of same length.")
    
#     n = len(parent1)
#     if n < 2:
#         return parent1
    
#     c = randint(1, n - 1)

#     return parent1[:c] + parent2[c:]


def crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Both parents must be of same length.")
    
    n = len(parent1)
    if n < 2:
        return parent1, parent2
    
    c = randint(1, n - 1)

    return parent1[:c] + parent2[c:], parent2[:c] + parent1[c:]


def mutation(genome, num=1, prob=0.5):
    for _ in range(num):
        index = randrange(len(genome))
        color = genome[index]
        otherColors = list(set(['r', 'g', 'b']) - set([color]))
        newColorIndex = randint(0, 1)
        newColor = otherColors[newColorIndex]
        genome[index] = genome[index] if random() > prob else newColor
    return genome


def fitness(genome, edges):
    fitnessCount = 0
    for i, vcolor in enumerate(genome):
        sameColor = False
        for edge in edges:
            if i == edge[0]:
                if vcolor == genome[edge[1]]:
                    sameColor = True
                    break
        
        if not sameColor:
            fitnessCount += 1
    return fitnessCount


def populationFitness(population):
    return sum([fitness(genome) for genome in population])


def selectionPair(population, edges):
    return choices(
        population=population,
        weights=[fitness(genome, edges) for genome in population],
        k=2
    )


def sortPopulation(population, edges):
    return sorted(population, key=lambda genome: fitness(genome, edges), reverse=True)


def geneticAlgorithm(
    edges, 
    populationSize, 
    genomeLength, 
    generationLimit, 
    fitnessLimit,
    maxFitnessGenerations
):

    startTime = time()
    bestFitness = 0
    fitnessCounter = 0
    fitnessValues = []

    population = generatePopulation(populationSize, genomeLength)

    for i in range(generationLimit):

        if time() - startTime >= 45:
            break

        population = sortPopulation(population, edges)
        currentFitness = fitness(population[0], edges)
        print(f'Generation {i}. Fitness Value = {currentFitness}')
        fitnessValues.append(currentFitness)

        if currentFitness <= bestFitness:
            fitnessCounter += 1
        else:
            fitnessCounter = 0
            bestFitness = currentFitness

        if currentFitness >= fitnessLimit or fitnessCounter >= maxFitnessGenerations:
            break

        newGeneration = population[:2]

        # for j in range(len(population) - 2):
        for j in range(int(len(population) / 2) - 1):
            parents = selectionPair(population, edges)

            # print(f'Fitness P1: {fitness(parents[0], edges)} P2: {fitness(parents[1], edges)}')

            # child = crossover(parents[0], parents[1])
            child1, child2 = crossover(parents[0], parents[1])
            # child = mutation(child)
            child1 = mutation(child1)
            child2 = mutation(child2)
            newGeneration += [child1, child2]

        population = newGeneration

    return population, i + 1, fitnessValues, time() - startTime


def printBestState(genome):
    printStr = 'Best state: '
    for i, vertex in enumerate(genome):
        printStr += f'{i}:{vertex.upper()} '
    print(printStr)


def main():
    gc = Graph_Creator()
    edges = gc.CreateGraphWithRandomEdges(800)
    # edges = gc.ReadGraphfromCSVfile("Testcases/200")
    print(f'No. of edges: {len(edges)}')

    populationSize = 100
    genomeLength = 50
    generationLimit = 50
    fitnessLimit = 50
    maxFitnessGenerations = 15

    population, generations, fitnessValues, timeTaken = geneticAlgorithm(
        edges,
        populationSize,
        genomeLength,
        generationLimit,
        fitnessLimit,
        maxFitnessGenerations
    )

    population = sortPopulation(population, edges)
    # print(f'Genome: {population[0]}')
    printBestState(population[0])
    finalFitness = fitness(population[0], edges)
    fitnessValues.append(finalFitness)
    print(f'Fitness Value: {finalFitness}')
    print(f'Generations required: {generations}')
    print(f'Time taken: {round(timeTaken, 2)}s')

    plt.title(f'Fitness across generations (#edges = {len(edges)})')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.plot(range(generations), fitnessValues[:generations])
    plt.show()


if __name__ == '__main__':
    main()