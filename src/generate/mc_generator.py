import numpy as np

class mc_generator_:
    def __init__(self, age, sex):
        """
        age: dict like {'0-17': count, '18-64': count, '65+': count}
        sex: dict like {'M': count, 'F': count}
        """
        self.age = age
        self.sex = sex

    def find_probs(self):
        # Normalize age and sex counts to get probabilities
        aa = np.array(list(self.age.values()), dtype=float)
        prob_age = aa / aa.sum()

        bb = np.array(list(self.sex.values()), dtype=float)
        prob_sex = bb / bb.sum()

        return prob_age, prob_sex

    def generate_samples(self, n_samples):
        prob_age, prob_sex = self.find_probs()

        # Sample age groups and sexes
        sampled_age_groups = np.random.choice(
            list(self.age.keys()), size=n_samples, p=prob_age)
        sampled_sexes = np.random.choice(
            list(self.sex.keys()), size=n_samples, p=prob_sex)

        # Convert age group to a specific age
        sampled_ages = []
        for age_group in sampled_age_groups:
            age_group = age_group.strip().replace('–', '-').replace('—', '-')
            if age_group == '18-64':
                sampled_ages.append(np.random.randint(18, 65))
            elif age_group == '0-17':
                sampled_ages.append(np.random.randint(0, 18))
            else:  # '65+'
                sampled_ages.append(np.random.randint(65, 100))

        # Stack into final array
        sampled_age_groups = np.array(sampled_age_groups).reshape(-1, 1)
        sampled_ages = np.array(sampled_ages).reshape(-1, 1)
        sampled_sexes = np.array(sampled_sexes).reshape(-1, 1)

        result = np.hstack([sampled_age_groups, sampled_ages, sampled_sexes])
        return result
