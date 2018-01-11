import math


def capacity_dist(cap1, cap2):
    frac = float(min(cap1, cap2)) / float(max(cap1, cap2))
    return (1.0 - frac)


def duration_dist(dur1, dur2, min_duration, max_duration):
    diff = abs(dur1.total_seconds() - dur2.total_seconds())
    return normalize(diff, min_duration, max_duration)


def users_dist(users1, users2):
    if len(users1) == 0 or len(users2) == 0:
        return 1
    num_similar = len(set(users1).intersection(users2))
    largest = float(max(len(users1), len(users2)))
    return ((largest - num_similar) / largest)


def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

seasons = {
    'winter': 0,
    'spring': 1,
    'summer': 2,
    'fall': 3
}


def season_dist(s1, s2):
    dx = float(abs(seasons[s1] - seasons[s2]))
    if dx > 2:
        dx = 4 - dx
    return (abs(dx) * 50.0) / 100.0
