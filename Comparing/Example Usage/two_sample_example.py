import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comparator import GroupComparator

comparator = GroupComparator()
sample1 = [5.1, 4.9, 5.0, 5.2, 5.1]
sample2 = [4.8, 4.9, 5.0, 5.1, 5.2]
t_stat, p_val = comparator.two_sample_ttest_independent(sample1, sample2, explain=True)
