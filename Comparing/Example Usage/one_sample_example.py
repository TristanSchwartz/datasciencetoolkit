import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comparator import GroupComparator

comparator = GroupComparator()

# Imagine this is a sample of exam scores of students in a class
sample = [78, 85, 92, 88, 76, 95, 83, 80, 89, 91]  # Sample of student exam scores (1-100 scale)
popmean = 85  # Population mean (e.g., average exam score for the class)

# Let's say we want to test if the average score of the sample is significantly different from the population mean (85).
t_stat, p_val = comparator.one_sample_ttest(sample, popmean, explain=True)

