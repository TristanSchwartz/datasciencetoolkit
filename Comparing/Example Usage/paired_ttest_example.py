import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comparator import GroupComparator

comparator = GroupComparator()
before = [5.1, 4.9, 5.0, 5.2, 5.1]
after = [5.0, 4.8, 4.9, 5.0, 5.0]
t_stat, p_val = comparator.paired_ttest(before, after, explain=True)
print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4f}")