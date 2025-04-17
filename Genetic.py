import random
import math

# --- Kelas untuk Individu (Kromosom) ---
class Individual:
    """Merepresentasikan satu individu dalam populasi."""
    def __init__(self, chr_length):
        self.chromosome = "".join(random.choice(['0', '1']) for _ in range(chr_length))
        self.fitness = None # Akan dihitung nanti
        self.decoded_values = None # Akan dihitung nanti

    def set_chromosome(self, chr_str):
        """Mengatur kromosom secara manual (misalnya setelah crossover/mutasi)."""
        self.chromosome = chr_str
        self.fitness = None # Reset fitness jika kromosom berubah
        self.decoded_values = None # Reset decoded values

    def __str__(self):
        """Representasi string dari individu."""
        return f"Chromosome: {self.chromosome}, Fitness: {self.fitness}"
    
# --- Kelas Utama Algoritma Genetika ---
class GeneticAlgorithm:
    """Mengelola proses Algoritma Genetika."""

    #menginisialisasi parameter GA dan state GA
    def __init__(self, pop_size, bits_per_var, n_vars, bounds, generations,
                 crossover_rate, mutation_rate, roulette_size, elitism_count):
        # Parameter GA
        self.pop_size = pop_size
        self.bits_per_var = bits_per_var
        self.n_vars = n_vars
        self.chr_length = bits_per_var * n_vars
        self.bounds = bounds
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.roullete_size = roulette_size
        self.elitism_count = elitism_count

        # State GA
        self.population = []
        self.best_individual = None
        self.best_overall_fitness = -float('inf')
        self.current_generation = 0

    # --- Proses Inisialisasi ---
    def initialize_population(self):
        """Membuat populasi awal."""
        self.population = [Individual(self.chr_length) for _ in range(self.pop_size)]
        print(f"Populasi awal dengan {len(self.population)} individu berhasil dibuat.")

    # --- Proses Dekode Kromosom ---
    def _decode_chromosome(self, chromosome_str):
        """Mendekode string kromosom menjadi nilai-nilai real [x1, x2]."""
        decoded_values = []
        max_int_value = (2**self.bits_per_var) - 1

        for i in range(self.n_vars):
            start_idx = i * self.bits_per_var
            end_idx = (i + 1) * self.bits_per_var
            binary_part = chromosome_str[start_idx:end_idx]
            int_value = int(binary_part, 2)
            min_bound, max_bound = self.bounds[i]
            real_value = min_bound + (int_value / max_int_value) * (max_bound - min_bound)
            decoded_values.append(real_value)
        return decoded_values

    # --- Proses Perhitungan Fitness ---
    def _calculate_fitness(self, individual):
        """Menghitung dan menyimpan fitness untuk satu individu."""
        if individual.fitness is not None: # Gunakan cache jika sudah dihitung
            return individual.fitness

        try:
            # Dekode jika belum
            if individual.decoded_values is None:
                 individual.decoded_values = self._decode_chromosome(individual.chromosome)
            x1, x2 = individual.decoded_values

            # Pastikan batas (redundant jika decode benar, tapi aman)
            x1 = max(self.bounds[0][0], min(self.bounds[0][1], x1))
            x2 = max(self.bounds[1][0], min(self.bounds[1][1], x2))

            # Fungsi fitness = -f(x1, x2)
            term1 = math.sin(x1) * math.cos(x2)
            angle_tan = x1 + x2
            if abs(math.cos(angle_tan)) < 1e-10:
                fitness_value = -float('inf') # kondisi jika tan tak terhingga
            else:
                term2 = math.tan(angle_tan)
                term3 = 0.75 * math.exp(1 - abs(x1))
                fitness_value = term1 * term2 + term3

            if math.isnan(fitness_value) or math.isinf(fitness_value):
                fitness_value = -float('inf') # kondisi hasil yang tidak valid

            individual.fitness = fitness_value
            return fitness_value

        except ValueError:
            individual.fitness = -float('inf') # Penalti untuk error perhitungan
            return -float('inf')

    # --- Proses Perhitungan fitness untuk semua individu dalam Populasi ---
    def _evaluate_population(self):
        """Menghitung fitness untuk semua individu dalam populasi."""
        current_best_fitness_in_gen = -float('inf')
        best_individual_in_gen = None
        for individual in self.population:
            fitness = self._calculate_fitness(individual)
            if fitness > current_best_fitness_in_gen:
                current_best_fitness_in_gen = fitness
                best_individual_in_gen = individual

        # Update best overall jika ditemukan yang lebih baik di generasi ini
        if best_individual_in_gen and current_best_fitness_in_gen > self.best_overall_fitness:
            self.best_overall_fitness = current_best_fitness_in_gen
            # Penting: salin individu terbaik, jangan hanya referensi
            self.best_individual = Individual(self.chr_length)
            self.best_individual.set_chromosome(best_individual_in_gen.chromosome)
            self.best_individual.fitness = best_individual_in_gen.fitness
            self.best_individual.decoded_values = best_individual_in_gen.decoded_values
            # Tampilkan peningkatan
            min_f_value = -self.best_overall_fitness
            print(f"Gen {self.current_generation+1}: Fitness Terbaik Baru = {self.best_overall_fitness:.5f} (f(x1,x2) ~ {min_f_value:.5f}), x1={self.best_individual.decoded_values[0]:.5f}, x2={self.best_individual.decoded_values[1]:.5f}")


    # --- Proses Pemilihan Orang Tua ---
    def _select_parents_roulette(self):
        """Memilih orang tua menggunakan seleksi roulette wheel."""
        # Hitung total fitness, perlu menangani fitness negatif
        min_fitness = min(ind.fitness for ind in self.population if ind.fitness is not None)
        # Jika ada fitness negatif, buat semua nilai positif dengan shifting
        offset = abs(min_fitness) + 1 if min_fitness < 0 else 0
        
        # Hitung total fitness yang sudah dishift
        total_fitness = sum(ind.fitness + offset for ind in self.population if ind.fitness is not None)
        
        selected_parents = []
        for _ in range(self.pop_size):
            # Jika total fitness 0, pilih secara acak
            if total_fitness <= 0:
                selected_parents.append(random.choice(self.population))
                continue
                
            # Pilih titik pada roulette wheel
            spin = random.uniform(0, total_fitness)
            current = 0
            
            # Temukan individu yang sesuai dengan titik spin
            for ind in self.population:
                if ind.fitness is not None:
                    current += (ind.fitness + offset)
                    if current >= spin:
                        selected_parents.append(ind)
                        break
            
            # Fallback jika loop selesai tanpa selection (rare case)
            if len(selected_parents) <= _:
                selected_parents.append(random.choice(self.population))
                
        return selected_parents

    # --- Proses Crossover (Pindah Silang) ---
    def _crossover(self, parent1, parent2):
        """Melakukan crossover antara dua kromosom orang tua."""
        chromo1 = parent1.chromosome
        chromo2 = parent2.chromosome
        child1_chromo, child2_chromo = chromo1, chromo2 # Default

        if random.random() < self.crossover_rate:
            point = random.randint(1, self.chr_length - 1)
            child1_chromo = chromo1[:point] + chromo2[point:]
            child2_chromo = chromo2[:point] + chromo1[point:]
        return child1_chromo, child2_chromo

    # --- Proses Mutasi ---
    def _mutate(self, chr_str):
        """Melakukan mutasi bit-flip pada string kromosom."""
        mutated_list = list(chr_str)
        for i in range(len(mutated_list)):
            if random.random() < self.mutation_rate:
                mutated_list[i] = '1' if mutated_list[i] == '0' else '0'
        return "".join(mutated_list)

    # --- Proses Pergantian Generasi ---
    def _evolve(self):
        """Menjalankan satu siklus evolusi (seleksi, crossover, mutasi)."""
        # 1. Evaluasi populasi saat ini (memastikan fitness terhitung & best overall terupdate)
        self._evaluate_population()

        next_population = []

        # 2. Elitisme
        # Urutkan populasi berdasarkan fitness (terbaik ke terburuk)
        sorted_population = sorted(self.population,
                                   key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf'),
                                   reverse=True)
        for i in range(self.elitism_count):
            # Salin individu elit ke populasi baru
            elite_ind = Individual(self.chr_length)
            elite_ind.set_chromosome(sorted_population[i].chromosome)
            elite_ind.fitness = sorted_population[i].fitness # Salin fitness juga
            elite_ind.decoded_values = sorted_population[i].decoded_values # Salin decoded juga
            next_population.append(elite_ind)


        # 3. Seleksi orang tua
        parents = self._select_parents_roulette()

        # 4. Crossover dan Mutasi untuk mengisi sisa populasi
        num_offspring_needed = self.pop_size - self.elitism_count
        offspring_count = 0
        while offspring_count < num_offspring_needed:
            p1, p2 = random.sample(parents, 2) # Ambil 2 orang tua acak

            child1_chromo, child2_chromo = self._crossover(p1, p2)

            mutated_child1_chromo = self._mutate(child1_chromo)
            mutated_child2_chromo = self._mutate(child2_chromo)

            # Buat individu baru untuk anak-anak
            child1 = Individual(self.chr_length)
            child1.set_chromosome(mutated_child1_chromo)
            child2 = Individual(self.chr_length)
            child2.set_chromosome(mutated_child2_chromo)

            if offspring_count < num_offspring_needed:
                next_population.append(child1)
                offspring_count += 1
            if offspring_count < num_offspring_needed:
                next_population.append(child2)
                offspring_count += 1

        # 5. Ganti populasi lama dengan yang baru
        self.population = next_population
        self.current_generation += 1

    # --- Menjalankan GA ---
    def run(self):
        """Menjalankan seluruh proses Algoritma Genetika."""
        print("Tugas kelompok Genetic Algorithm")
        print(f"Ukuran Populasi: {self.pop_size}, Generasi: {self.generations}")
        print(f"Panjang Kromosom: {self.chr_length} ({self.bits_per_var} bits/variabel)")
        print("-" * 30)

        self.initialize_population()

        # Loop Generasi
        while self.current_generation < self.generations:
            self._evolve()

        # Evaluasi populasi terakhir untuk memastikan best_individual terupdate
        self._evaluate_population()
        print("-" * 30)
        
    # --- Mendapatkan Hasil ---
    def get_best_solution(self):
        """Mengembalikan individu terbaik yang ditemukan."""
        if self.best_individual:
            # Pastikan nilai dekode ada
            if self.best_individual.decoded_values is None:
                 self.best_individual.decoded_values = self._decode_chromosome(self.best_individual.chromosome)
            # Nilai minimum f(x1, x2) = - fitness maksimum
            min_objective_value = -self.best_individual.fitness if self.best_individual.fitness is not None else float('nan')

            return {
                "chromosome": f"[{', '.join(map(str, self.best_individual.chromosome))}]",
                "fitness": self.best_individual.fitness,
                "min_f_value": min_objective_value,
                "x1": self.best_individual.decoded_values[0],
                "x2": self.best_individual.decoded_values[1]
            }
        else:
            return None

# --- Parameter dan Eksekusi ---
if __name__ == "__main__":
    # Parameter Algoritma Genetika
    POP_SIZE = 50
    BITS_PER_VAR = 10
    N_VARIABLES = 2
    BOUNDS = [(-10.0, 10.0), (-10.0, 10.0)]
    GENERATIONS = 20
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.05
    ROULETTE_SIZE = 5
    ELITISM_COUNT = 1

    # Buat objek GA
    total_init = GeneticAlgorithm(
        pop_size=POP_SIZE,
        bits_per_var=BITS_PER_VAR,
        n_vars=N_VARIABLES,
        bounds=BOUNDS,
        generations=GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        roulette_size=ROULETTE_SIZE,
        elitism_count=ELITISM_COUNT
    )

    # Jalankan GA
    total_init.run()

    # Dapatkan dan tampilkan solusi terbaik
    best_solution = total_init.get_best_solution()

    if best_solution:
        print("\n--- Output Program ---")
        print(f"Kromosom terbaik ditemukan  : {best_solution['chromosome']}")
        print(f"Nilai Fitness Maksimum      : {best_solution['fitness']:.8f}")
        print(f"Nilai Minimum f(x1, x2)     : {best_solution['min_f_value']:.8f}")
        print(f"Nilai x1 hasil dekode       : {best_solution['x1']:.8f}")
        print(f"Nilai x2 hasil dekode       : {best_solution['x2']:.8f}")
    else:
        print("\nTidak ditemukan solusi yang valid.")