from pathlib import Path
from heapq import nlargest
from random import random, randint, choices, choice

from const import *


class Chromosome:
    """
    Represents a single chromosome as a word

    :param x: row number of the word’s first symbol
    :param y: column number of the word’s first symbol
    :param direction: horizontal (0) or vertical (1)
    """

    def __init__(self, x: int, y: int, direction: int):
        self.x, self.y = x, y
        self.direction = direction

    def clone(self):
        return Chromosome(self.x, self.y, self.direction)


class Individual(list):
    """
    Represents a single individual as a crossword

    :param chromosomes: list of word chromosomes
    """

    def __init__(self, chromosomes: list[Chromosome]):
        super().__init__(chromosomes)
        self.fitness: tuple[int, int] = 0, 0

    def clone(self):
        return Individual(list(self))


class CrosswordGrid:
    """
    Represents a crossword grid

    :param individual: crossword Individual
    :param words: list of words
    """

    class Word:
        """
        Represents a word on a crossword grid

        :param coordinates: coordinates of the word’s first symbol
        :param word: word on a crossword
        :param direction: horizontal (0) or vertical (1)
        """

        def __init__(self, coordinates: tuple[int, int], word: str, direction: int):
            self.start = coordinates
            self.end = (coordinates[0] + (not direction) * (len(word) - 1),
                        coordinates[1] + direction * (len(word) - 1))
            self.direction = direction
            self.text = word

    def __init__(self, individual: Individual, words: list[str]):
        self.individual = individual
        self.words = [
            self.Word((chromo.x, chromo.y), word, chromo.direction)
            for chromo, word in zip(individual, words)
        ]

    def __str__(self):
        grid = self._get_visual_grid()
        return '   ' + ' '.join(
            map(lambda c: ' ' * (2 - len(str(c))) + str(c), range(GRID_WIDTH))
        ) + '\n' + '\n'.join(
            ' ' * (2 - len(str(i))) + str(i) + '  ' + '  '.join(
                grid[i][j][0]
                for j in range(GRID_WIDTH)
            ) for i in range(GRID_HEIGHT)
        )

    def _get_intersections_graph(self) -> dict[int: list[int]]:
        intersections: dict[int: set[int]] = {i: set() for i in range(len(self.words))}
        for i in range(len(self.words)):
            for j in range(i + 1, len(self.words)):
                if (self.words[i].direction and not self.words[j].direction and
                    self.words[j].start[1] in range(self.words[i].start[1], self.words[i].end[1] + 1) and
                    self.words[i].start[0] in range(self.words[j].start[0], self.words[j].end[0] + 1)) or \
                        (not self.words[i].direction and self.words[j].direction and
                         self.words[i].start[1] in range(self.words[j].start[1], self.words[j].end[1] + 1) and
                         self.words[j].start[0] in range(self.words[i].start[0], self.words[i].end[0] + 1)):
                    intersections[i].add(j)
                    intersections[j].add(i)
        return intersections

    def _get_visual_grid(self) -> list[list[tuple[str, int]]]:
        grid = [[(' ', -1)] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        for word in self.words:
            for i, char in enumerate(word.text):
                x, y = word.start[0] + i * (not word.direction), word.start[1] + i * word.direction

                grid[y][x] = (
                    char if grid[y][x][0] in (char, ' ') else '*',
                    word.direction if grid[y][x][0] == ' ' or grid[y][x][1] == word.direction else 2
                )
        return grid

    def get_out_of_bounds_count(self) -> int:
        out_of_bound_count = 0
        for word in self.words:
            if word.end[0 if word.direction == 0 else 1] > (GRID_WIDTH if word.direction == 0 else GRID_HEIGHT):
                out_of_bound_count += 1
                continue
        return out_of_bound_count

    def get_neighbours_count(self) -> int:
        neighbours_count = 0
        grid = self._get_visual_grid()
        for word in self.words:
            neighbours_count += (0 <= word.start[1] - word.direction and
                                 0 <= word.start[0] - (not word.direction) and
                                 grid[word.start[1] - word.direction][word.start[0] - (not word.direction)][0] != ' ')

            neighbours_count += (word.end[1] + word.direction < GRID_HEIGHT and
                                 word.end[0] + (not word.direction) < GRID_WIDTH and
                                 grid[word.end[1] + word.direction][word.end[0] + (not word.direction)][0] != ' ')

            for i, char in enumerate(word.text):
                x, y = word.start[0] + i * (not word.direction), word.start[1] + i * word.direction

                if grid[y][x][1] != 2:
                    if word.direction == 0:
                        if y - 1 < 0 or GRID_HEIGHT <= y + 1:
                            break
                        neighbours_count += grid[y + 1][x][0] != ' ' or grid[y - 1][x][0] != ' '
                    else:
                        if x - 1 < 0 or GRID_WIDTH <= x + 1:
                            break
                        neighbours_count += grid[y][x + 1][0] != ' ' or grid[y][x - 1][0] != ' '

        return neighbours_count

    def get_intersections_count(self) -> int:
        return sum(len(x) for x in self._get_intersections_graph().values()) // 2

    def get_overlaps_count(self) -> int:
        return [cell[0] for cell in sum(self._get_visual_grid(), start=[])].count('*')

    def get_disconnected_words_count(self) -> int:
        def dfs(a):
            visited.add(a)
            for b in intersections[a]:
                if b not in visited:
                    dfs(b)

        intersections = self._get_intersections_graph()
        visited = set()
        dfs(list(intersections.keys())[0])

        return len(intersections) - len(visited)


def get_random_chromosome(word: str):
    direction = randint(0, 1)
    return Chromosome(randint(0, GRID_WIDTH - (not direction) * (len(word) - 1) - 1),
                      randint(0, GRID_HEIGHT - direction * (len(word) - 1) - 1),
                      direction)


def get_random_individual(words: list[str]):
    return Individual([get_random_chromosome(word) for word in words])


def get_random_population(population_size: int, words: list[str]):
    return [get_random_individual(words) for _ in range(population_size)]


def crossword(words: list[str],
              population_size: int, max_generations: int,
              p_mutation: float, p_crossover: float) -> Individual:
    """
    Generates a crossword using an evolutionary algorithm

    :param words: list of words
    :param population_size: size of populations
    :param max_generations: maximal count of generations
    :param p_mutation: mutation probability
    :param p_crossover: crossover probability

    :return the desired crossword as an Individual
    """

    def fitness(ind: Individual) -> tuple[int, int]:
        reward, penalty = 0, 0
        grid = CrosswordGrid(ind, words)

        intersections = grid.get_intersections_count()
        penalty -= (len(words) - 1 - intersections) * CROSSING_PENALTY if intersections < len(words) - 1 else 0
        reward += CROSSING_REWARD if intersections >= len(words) - 1 else 0

        disconnected_words = grid.get_disconnected_words_count()
        penalty -= disconnected_words * CONNECTIVITY_PENALTY if disconnected_words > 0 else 0
        reward += CONNECTIVITY_REWARD if disconnected_words == 0 else 0

        overlaps = grid.get_overlaps_count()
        penalty -= overlaps * OVERLAPPING_PENALTY if overlaps > 0 else 0
        reward += OVERLAPPING_REWARD if overlaps == 0 else 0

        neighbours = grid.get_neighbours_count()
        penalty -= neighbours * NEIGHBOURS_PENALTY if neighbours > 0 else 0
        reward += NEIGHBOURS_REWARD if neighbours == 0 else 0

        return reward, penalty

    def crossover(parent1: Individual, parent2: Individual) -> Individual:
        return Individual([choice(chromos).clone() for chromos in zip(parent1, parent2)])

    def mutate(ind: Individual) -> Individual:
        mutated_ind: Individual = ind.clone()

        for i in choices(range(len(ind)),
                         k=choices(
                             range(1, len(ind) + 1),
                             weights=range(len(ind), 0, -1),
                             k=1)[0]
                         ):
            mutated_ind[i] = get_random_chromosome(words[i])

        return mutated_ind

    population = get_random_population(population_size, words)
    for _ in range(max_generations):
        new_population = [
            crossover(
                *nlargest(2,
                          choices(population, k=len(population) // 2),
                          lambda ind: sum(ind.fitness))
            ) if random() < p_crossover else population[i] for i in range(len(population))
        ]

        for i in range(len(population)):
            if random() < p_mutation:
                new_population[i] = mutate(new_population[i])

        population = new_population

        for i in range(len(population)):
            population[i].fitness = fitness(population[i])
            if population[i].fitness[1] == 0:
                return population[i]

    return max(population, key=lambda ind: sum(ind.fitness))


def main():
    inputs_dir, outputs_dir = Path('inputs'), Path('outputs')
    outputs_dir.mkdir(exist_ok=True)

    for input_file in sorted(inputs_dir.glob('input*.txt'),
                             key=lambda f: int(f.name.split('input')[1].split('.txt')[0])):
        m = int(input_file.name.split('input')[1].split('.txt')[0])
        words = open(input_file).read().split('\n')[:-1]

        try:
            cw = crossword(words,
                           POPULATION_SIZE,
                           MAX_GENERATIONS,
                           MUTATION_PROBABILITY,
                           CROSSOVER_PROBABILITY)

            print(f'# {m}\n______')
            print(CrosswordGrid(cw, words), '\n')

            open(outputs_dir / f'output{m}.txt', 'w').write(
                '\n'.join([f'{w.x} {w.y} {w.direction}' for w in cw]) + '\n'
            )
        except Exception as E:
            open(outputs_dir / f'output{m}.txt', 'w').write(str(E))


if __name__ == '__main__':
    main()
