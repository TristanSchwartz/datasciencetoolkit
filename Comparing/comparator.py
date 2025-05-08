import numpy as np
from typing import Tuple, Union, List
from math import sqrt
from scipy.stats import t, f

class GroupComparator:
    def one_sample_ttest(
            self, 
            sample: Union[List[float], np.ndarray], 
            popmean: float, 
            explain:bool = False 
            ) -> Tuple[float, float]:
        """
        Perform a one-sample t-test. Used to compare a sample mean against a known population mean.

        Args:
            sample (Union[List[float], np.ndarray]): Sample data.
            popmean (float): Population mean to compare against.

        Returns:
            Tuple[float, float]: t-statistic and p-value.
        """
        sample = np.array(sample)
        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)

        t_stat = (sample_mean - popmean) / (sample_std / sqrt(n))
        p_val = 2 * (1 - t.cdf(abs(t_stat), df=n-1))

        print(f"[One-Sample T-Test] Sample mean = {sample_mean:.4f}, Population mean = {popmean}")
        print(f"t = {t_stat:.4f}, p = {p_val:.4f}")
        
        if explain:
            print(
                "\nExplanation:"
                f"\n- The t-statistic of {t_stat:.4f} indicates how many standard errors the sample mean is away from the population mean."
                f"\n- The p-value of {p_val:.4f} represents the probability of observing a sample mean this extreme assuming the population mean is correct."
                f"\n- Since the p-value is {'less' if p_val < 0.05 else 'greater'} than 0.05, we "
                + ("reject" if p_val < 0.05 else "fail to reject")
                + " the null hypothesis — "
                + ("suggesting a statistically significant difference from the population mean."
                if p_val < 0.05 else "there is not enough evidence to conclude a difference from the population mean.")
            )
        return t_stat, p_val

    def two_sample_ttest_independent(
            self,
            sample1: Union[List[float], np.ndarray],
            sample2: Union[List[float], np.ndarray],
            equal_var=True,
            explain: bool = False
        ) -> Tuple[float, float]:
        """
        Perform an independent two-sample t-test. Used to compare means from two different groups to see if they are significantly different.

        Args:
            sample1 (Union[List[float], np.ndarray]): First sample data.
            sample2 (Union[List[float], np.ndarray]): Second sample data.
            equal_var (bool): Assume equal population variances.
            explain (bool): Whether to print detailed explanation of the result.

        Returns:
            Tuple[float, float]: t-statistic and p-value.
        """
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        n1 = len(sample1)
        n2 = len(sample2)
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        std1 = np.std(sample1, ddof=1)
        std2 = np.std(sample2, ddof=1)

        if equal_var:
            pooled_var = (((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            se = sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            se = sqrt((std1**2 / n1) + (std2**2 / n2))
            df_num = (std1**2 / n1 + std2**2 / n2)**2
            df_denom = ((std1**2 / n1)**2 / (n1 - 1)) + ((std2**2 / n2)**2 / (n2 - 1))
            df = df_num / df_denom

        t_stat = (mean1 - mean2) / se
        from scipy.stats import t
        p_val = 2 * (1 - t.cdf(abs(t_stat), df=df))

        print(f"[Two-Sample T-Test] Mean1 = {mean1:.4f}, Mean2 = {mean2:.4f}, Equal variances assumed: {equal_var}")
        print(f"t = {t_stat:.4f}, p = {p_val:.4f}")

        if explain:
            print(
                "\nExplanation:"
                f"\n- The t-statistic of {t_stat:.4f} quantifies how far apart the sample means are in terms of standard error."
                f"\n- The p-value of {p_val:.4f} is the probability of observing such a difference between sample means assuming the true population means are equal."
                f"\n- Since the p-value is {'less' if p_val < 0.05 else 'greater'} than 0.05, we "
                + ("reject" if p_val < 0.05 else "fail to reject")
                + " the null hypothesis — "
                + ("suggesting a statistically significant difference between the group means."
                if p_val < 0.05 else "indicating there is not enough evidence to conclude the group means are different.")
            )

        return t_stat, p_val

    def paired_ttest(
            self,
            sample1: Union[List[float], np.ndarray],
            sample2: Union[List[float], np.ndarray],
            explain: bool = False
        ) -> Tuple[float, float]:
        """
        Perform a paired sample t-test. This test compares the means of two related samples — typically 'before' and 'after' measurements on the same subjects — to determine if there is a statistically significant difference.

        Args:
            sample1 (Union[List[float], np.ndarray]): Measurements before intervention.
            sample2 (Union[List[float], np.ndarray]): Measurements after intervention.
            explain (bool): Whether to print detailed explanation of the result.

        Returns:
            Tuple[float, float]: t-statistic and p-value.
        """
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        diff = sample1 - sample2
        n = len(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        se = std_diff / sqrt(n)
        t_stat = mean_diff / se
        from scipy.stats import t
        p_val = 2 * (1 - t.cdf(abs(t_stat), df=n - 1))

        print(f"[Paired T-Test] Mean difference = {mean_diff:.4f}")
        print(f"t = {t_stat:.4f}, p = {p_val:.4f}")

        if explain:
            print(
                "\nExplanation:"
                f"\n- The t-statistic of {t_stat:.4f} measures how far the average difference between pairs is from zero, relative to the variability in the differences."
                f"\n- The p-value of {p_val:.4f} tells us the probability of observing such a mean difference if there were truly no difference."
                f"\n- Since the p-value is {'less' if p_val < 0.05 else 'greater'} than 0.05, we "
                + ("reject" if p_val < 0.05 else "fail to reject")
                + " the null hypothesis — "
                + ("indicating a statistically significant change between the paired samples."
                if p_val < 0.05 else "indicating the change between the paired samples is not statistically significant.")
            )

        return t_stat, p_val

    def anova(
            self, 
            *groups: List[float], 
            explain: bool = False
            ) -> Tuple[float, float]:
        """
        Perform a one-way ANOVA test. Used to determine whether there are statistically significant differences between the means of three or more independent groups. 
        Commonly applied in experiments or observational studies where one categorical factor is tested across multiple levels (e.g., comparing test scores across different teaching methods).

        Args:
            *groups (List[float]): Two or more groups of sample data.
            explain (bool): If True, prints a detailed explanation of the results.

        Returns:
            Tuple[float, float]: F-statistic and p-value.
        """
        groups = [np.array(g) for g in groups]
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        grand_mean = np.mean([val for group in groups for val in group])

        # Calculate sum of squares between groups (SS_between)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        df_between = k - 1  # Degrees of freedom between groups

        # Calculate sum of squares within groups (SS_within)
        ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in groups)
        df_within = n_total - k  # Degrees of freedom within groups

        # Mean squares (MS)
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        # Calculate F-statistic
        f_stat = ms_between / ms_within

        # Calculate p-value from the F-distribution
        p_val = 1 - f.cdf(f_stat, df_between, df_within)

        group_means = [np.mean(g) for g in groups]
        print(f"[ANOVA] Group means = {[round(m, 4) for m in group_means]}")
        print(f"F = {f_stat:.4f}, p = {p_val:.4f}")

        # Explanation of results
        if explain:
            print("\nExplanation:")
            print("1. The **F-statistic** represents the ratio of the variance between the groups to the variance within the groups.")
            print("   - A larger F-statistic indicates a greater difference between group means compared to within-group variability.")
            print("   - A small F-statistic suggests that the group means are relatively similar to each other.")
            
            print("2. The **p-value** tells us whether the observed difference is statistically significant.")
            print("   - If the p-value is below 0.05 (typically), we reject the null hypothesis, suggesting that at least one group mean is different.")
            print("   - If the p-value is above 0.05, we fail to reject the null hypothesis, indicating no significant difference between the group means.")
            
            if p_val < 0.05:
                print("   - **Result**: Statistically significant differences between groups.")
            else:
                print("   - **Result**: No statistically significant differences between groups.")

        else:
            print("Result: " + ("Statistically significant differences between groups."
                                if p_val < 0.05 else "No statistically significant differences between groups."))

        return f_stat, p_val
