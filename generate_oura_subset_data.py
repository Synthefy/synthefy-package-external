#!/usr/bin/env python3
"""
Generate sample CSV data for oura_subset dataset with 10,000 rows.
"""

import csv
import random
import math

# Set random seed for reproducibility
random.seed(42)


def normal_distribution(mean: float, std: float) -> float:
    """Generate a value from a normal distribution using Box-Muller transform."""
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z0 * std + mean


def exponential_distribution(lam: float) -> float:
    """Generate a value from an exponential distribution."""
    return -math.log(1.0 - random.random()) / lam

# Number of users and entries per user
NUM_USERS = 10
ENTRIES_PER_USER = 10000
NUM_ROWS = NUM_USERS * ENTRIES_PER_USER

# Column definitions
GROUP_LABELS = ["user_id"]
TIMESERIES = ["average_hrv", "lowest_heart_rate", "age_cva_diff"]
DISCRETE = ["gender_male"]
CONTINUOUS = [
    "awake_mins",
    "age",
    "low_activity_time",
    "deep_mins",
    "sleep_duration",
    "restored_duration",
    "non_wear_time",
    "rem_mins",
    "steps",
    "active_calories",
]

# All columns in order
ALL_COLUMNS = GROUP_LABELS + TIMESERIES + DISCRETE + CONTINUOUS


def get_user_id(row_index: int) -> str:
    """Get user ID based on row index (each user has ENTRIES_PER_USER entries)."""
    user_index = ((row_index - 1) // ENTRIES_PER_USER) + 1
    return f"user_{user_index:05d}"


def generate_average_hrv() -> float:
    """Generate average HRV (Heart Rate Variability) in ms."""
    return round(normal_distribution(45, 8), 1)


def generate_lowest_heart_rate() -> int:
    """Generate lowest heart rate in bpm."""
    return int(normal_distribution(60, 5))


def generate_age_cva_diff() -> int:
    """Generate age CVA difference."""
    return int(normal_distribution(5, 2))


def generate_gender_male() -> int:
    """Generate gender (1 for male, 0 for female)."""
    return random.randint(0, 1)


def generate_awake_mins() -> int:
    """Generate awake minutes."""
    return int(normal_distribution(410, 30))


def generate_age() -> int:
    """Generate age."""
    return int(normal_distribution(35, 8))


def generate_low_activity_time() -> int:
    """Generate low activity time in minutes."""
    return int(normal_distribution(170, 25))


def generate_deep_mins() -> int:
    """Generate deep sleep minutes."""
    return int(normal_distribution(95, 15))


def generate_sleep_duration() -> int:
    """Generate total sleep duration in minutes."""
    return int(normal_distribution(470, 30))


def generate_restored_duration() -> int:
    """Generate restored duration in minutes."""
    # Usually close to sleep duration
    sleep = generate_sleep_duration()
    return int(sleep * random.uniform(0.85, 1.0))


def generate_non_wear_time() -> int:
    """Generate non-wear time in minutes."""
    # Most people have 0, some have a bit
    if random.random() < 0.1:  # 10% chance of non-wear
        return int(exponential_distribution(1.0 / 20))
    return 0


def generate_rem_mins() -> int:
    """Generate REM sleep minutes."""
    return int(normal_distribution(110, 15))


def generate_steps() -> int:
    """Generate daily steps."""
    return int(normal_distribution(8500, 1500))


def generate_active_calories() -> int:
    """Generate active calories."""
    return int(normal_distribution(350, 60))


# Cache for user-specific attributes (age, gender) that should be consistent per user
user_attributes_cache = {}


def get_user_attributes(user_id: str):
    """Get or create cached attributes for a user (age, gender)."""
    if user_id not in user_attributes_cache:
        user_attributes_cache[user_id] = {
            "age": generate_age(),
            "gender_male": generate_gender_male(),
        }
    return user_attributes_cache[user_id]


# Mapping of column names to generator functions
GENERATORS = {
    "user_id": lambda i: get_user_id(i),
    "average_hrv": lambda i: generate_average_hrv(),
    "lowest_heart_rate": lambda i: generate_lowest_heart_rate(),
    "age_cva_diff": lambda i: generate_age_cva_diff(),
    "gender_male": lambda i: get_user_attributes(get_user_id(i))["gender_male"],
    "awake_mins": lambda i: generate_awake_mins(),
    "age": lambda i: get_user_attributes(get_user_id(i))["age"],
    "low_activity_time": lambda i: generate_low_activity_time(),
    "deep_mins": lambda i: generate_deep_mins(),
    "sleep_duration": lambda i: generate_sleep_duration(),
    "restored_duration": lambda i: generate_restored_duration(),
    "non_wear_time": lambda i: generate_non_wear_time(),
    "rem_mins": lambda i: generate_rem_mins(),
    "steps": lambda i: generate_steps(),
    "active_calories": lambda i: generate_active_calories(),
}


def generate_row(index: int) -> list:
    """Generate a single row of data."""
    return [GENERATORS[col](index) for col in ALL_COLUMNS]


def main():
    """Generate the CSV file."""
    output_file = "oura_subset_sample_data.csv"

    print(f"Generating {NUM_USERS} users with {ENTRIES_PER_USER} entries each ({NUM_ROWS} total rows)...")

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(ALL_COLUMNS)

        # Generate and write rows
        for i in range(1, NUM_ROWS + 1):
            row = generate_row(i)
            writer.writerow(row)

            # Progress indicator
            if i % 1000 == 0:
                user_num = ((i - 1) // ENTRIES_PER_USER) + 1
                print(f"Generated {i} rows ({user_num} users complete)...")

    print(f"✓ Successfully generated {NUM_ROWS} rows ({NUM_USERS} users × {ENTRIES_PER_USER} entries) in {output_file}")


if __name__ == "__main__":
    main()
