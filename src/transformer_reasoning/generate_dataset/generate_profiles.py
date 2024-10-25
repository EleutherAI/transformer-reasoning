import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Union
import numpy as np
from datasets import Dataset, Features, Value, Sequence
from transformer_reasoning.utils import get_project_root
import argparse

with open(get_project_root() / "generated_data/NameDatabases/NamesDatabases/first names/us.txt", "r") as file:
    FIRST_NAMES = file.read().splitlines()
MIDDLE_NAMES = FIRST_NAMES

with open(get_project_root() / "generated_data/NameDatabases/NamesDatabases/surnames/us.txt", "r") as file:
    LAST_NAMES = file.read().splitlines()

with open(get_project_root() / "generated_data/towns/aus_towns.txt", "r") as file:
    CITIES = file.read().splitlines()

with open(get_project_root() / "generated_data/universities/chn_univs.txt", "r") as file:
    UNIVERSITIES = file.read().splitlines()


with open(get_project_root() / "generated_data/employers/ind_employ.txt", "r") as file:
    EMPLOYERS = file.read().splitlines()

used_names = set()

def generate_unique_name() -> Tuple[str, str, str]:
    global used_names

    while True:
        first = random.choice(FIRST_NAMES)
        middle = random.choice(MIDDLE_NAMES)
        last = random.choice(LAST_NAMES)
        if (first, middle, last) not in used_names:
            used_names.add((first, middle, last))
            return first, middle, last

def generate_birthdate() -> datetime:
    start_date = datetime(1900, 1, 1)
    end_date = datetime(2099, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_number_of_days)
    return random_date.replace(day=min(random_date.day, 28))

def generate_all_names(n: int) -> List[str]:
    names = []
    for _ in range(n):
        first, middle, last = generate_unique_name()
        names.append(f"{first} {middle} {last}")
    return names

def partition_names(names: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    random.shuffle(names)
    half = len(names) // 2
    parents = names[:half]
    children = names[half:]
    
    return parents, children, names.copy(), names.copy()

def create_relationships(parents: List[str], children: List[str], friend_pool: List[str], enemy_pool: List[str]) -> Tuple[Dict[str, Dict[str, Dict[str, Union[str, int]]]], Dict[str, int]]:
    relationships = {}
    name_to_index = {}
    
    for i, name in enumerate(parents + children):
        name_to_index[name] = i
        relationships[name] = {
            "parent": {"name": "", "index": -1},
            "child": {"name": "", "index": -1},
            "best_friend": {"name": "", "index": -1},
            "worst_enemy": {"name": "", "index": -1}
        }
    
    for parent, child in zip(parents, children):
        relationships[parent]["child"] = {"name": child, "index": name_to_index[child]}
        relationships[child]["parent"] = {"name": parent, "index": name_to_index[parent]}
    
    random.shuffle(friend_pool)
    random.shuffle(enemy_pool)
    
    for i in range(0, len(friend_pool), 2):
        friend1, friend2 = friend_pool[i], friend_pool[i+1]
        relationships[friend1]["best_friend"] = {"name": friend2, "index": name_to_index[friend2]}
        relationships[friend2]["best_friend"] = {"name": friend1, "index": name_to_index[friend1]}
    
    for i in range(0, len(enemy_pool), 2):
        enemy1, enemy2 = enemy_pool[i], enemy_pool[i+1]
        relationships[enemy1]["worst_enemy"] = {"name": enemy2, "index": name_to_index[enemy2]}
        relationships[enemy2]["worst_enemy"] = {"name": enemy1, "index": name_to_index[enemy1]}
    
    return relationships, name_to_index

def generate_profiles():
    global N
    all_names = generate_all_names(N)
    parents, children, friend_pool, enemy_pool = partition_names(all_names)
    relationships, name_to_index = create_relationships(parents, children, friend_pool, enemy_pool)

    for name in all_names:
        profile = {
            "name": name,
            "index": name_to_index[name],
            "birth_date": generate_birthdate(),
            "birth_city": random.choice(CITIES),
            "university": random.choice(UNIVERSITIES),
            "employer": random.choice(EMPLOYERS),
            "parent": relationships[name]["parent"],
            "child": relationships[name]["child"],
            "best_friend": relationships[name]["best_friend"],
            "worst_enemy": relationships[name]["worst_enemy"],
        }
        yield profile

# Update the features to include new fields
chosen_params = Features({
    'name': Value('string'),
    'index': Value('int32'),
    'birth_date': Value('timestamp[s]'),
    'birth_city': Value('string'),
    'university': Value('string'),
    'employer': Value('string'),
    'parent': {
        'name': Value('string'),
        'index': Value('int32')
    },
    'child': {
        'name': Value('string'),
        'index': Value('int32')
    },
    'best_friend': {
        'name': Value('string'),
        'index': Value('int32')
    },
    'worst_enemy': {
        'name': Value('string'),
        'index': Value('int32')
    },
    'bio': Value('string')
})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate profiles dataset")
    parser.add_argument("N", type=int, help="Number of profiles to generate")
    args = parser.parse_args()

    N = args.N
    dataset = Dataset.from_generator(generate_profiles, features=chosen_params)
    dataset.save_to_disk(str(get_project_root() / f"generated_data/profiles_dataset_{N}"))
