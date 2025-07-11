import numpy as np


class IPF: 
    def __init__(self, age_dict, sex_dict, tol=1e-5, max_iter=1000):
        # Store marginal distributions and label mappings
        self.age_keys = list(age_dict.keys())
        self.sex_keys = list(sex_dict.keys())
        self.age = np.array(list(age_dict.values()))
        self.sex = np.array(list(sex_dict.values()))
        
        self.tol = tol
        self.max_iter = max_iter
        self.mat_seed = None
        self.fitted_matrix = None
        self.rng = np.random.default_rng(seed=42)

    def seed_gen(self): # Seed matrix generation method
        n_row = len(self.age)
        n_col = len(self.sex)
        avg = np.average([np.average(self.age), np.average(self.sex)])
        low_val = int(avg * 0.5)
        high_val = int(avg * 1.5)
        self.mat_seed = self.rng.integers(low=low_val, high=high_val, size=(n_row, n_col))
        return self.mat_seed

    def row_adjust(self, matrix): # Adjust and scale rows
        sf = self.age / matrix.sum(axis=1)
        return (matrix.T * sf).T

    def column_adjust(self, matrix):  # Adjust and scale columns
        sf = self.sex / matrix.sum(axis=0)
        return matrix * sf

    def fit(self): # Fit function to generate final matrix structure
        matrix = self.seed_gen().astype(float)
        for _ in range(self.max_iter):
            prev = matrix.copy()
            matrix = self.row_adjust(matrix)
            matrix = self.column_adjust(matrix)
            if np.linalg.norm(matrix - prev, ord='fro') < self.tol: # Limit convergence using Frobenious norm
                print(f"Matrix converged on {_} iterations")
                break
        self.fitted_matrix = matrix
        return matrix

    def generate(self, n_samples=1000, random_state=42): # Genetate ynthetic population
        if self.fitted_matrix is None:
            raise ValueError("You must call .fit() before generating samples.")
        
        prob_matrix = self.fitted_matrix / self.fitted_matrix.sum() # Change matrix to probability matrix
        flattened_probs = prob_matrix.flatten() # Flatten matrix

        # Map index positions to actual labels
        joint_labels = [(self.age_keys[i], self.sex_keys[j])
                        for i in range(prob_matrix.shape[0])
                        for j in range(prob_matrix.shape[1])] # Combination of age,sex fewatures 

        rng = np.random.default_rng(seed=random_state)
        indices = rng.choice(len(joint_labels), size=n_samples, p=flattened_probs) # Generate synthetioc data stratified 
        samples = [joint_labels[i] for i in indices]

        samp_array = np.array(samples, dtype=str)

        # Generate synthetic age based on age group
        samp_array = np.array(samples, dtype=str)

        synthetic_ages = []
        for row in samp_array:
            age_group = row[0].strip().replace('–', '-').replace('—', '-')  # Normalize
            if age_group == '18-64':
                synthetic_ages.append(np.random.randint(18, 65))
            elif age_group == '0-17':
                synthetic_ages.append(np.random.randint(0, 18))
            else:
                synthetic_ages.append(np.random.randint(65, 100))

        synthetic_ages = np.array(synthetic_ages).reshape(-1, 1)
        result = np.hstack([samp_array, synthetic_ages])

        # print(result)
        return result
