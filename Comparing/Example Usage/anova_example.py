import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comparator import GroupComparator

comparator = GroupComparator()
# Imagine These are the test scores of three different groups of students
group1 = [95, 90, 92] # Fully in person
group2 = [83, 88, 85] # Fully online
group3 = [84, 89, 86] # Hybrid

#Let's say we want to compare the means of these three groups to see if there's a significant difference in their test scores.
f_stat, p_val = comparator.anova(group1, group2, group3, explain=True)
