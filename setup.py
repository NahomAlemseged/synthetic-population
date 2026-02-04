from setuptools import find_packages,setup
from typing import List


def get_requirements()->List[str]:
    '''
    function returing list of requirements

    '''
    requirement_list:list[str] = []    
    try:
        with open("requirements.txt",'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()

                # ignore empty lines and -e .

                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_list                


# print(get_requirements())
setup(
    name="Synthetic data generator",
    version='0.0.1',
    author="Bi lab",
    author_email="Nahom.A.Worku@uth.tmc.edu",
    packages=find_packages(),
    install_requires = get_requirements()
)