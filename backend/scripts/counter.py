import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FrequencyCounter:
    """
    Class to analyze the frequency of topics over time based on a DataFrame.
    """

    def __init__(self, df) -> None:
        """
        Initialize the FrequencyCounter.

        Args:
            df (pandas.DataFrame): DataFrame containing data with columns "publicationdateyear" and "subject".
        """
        self.df = df
        self.subject_df = df[["publicationdateyear", "subject"]].dropna()
        self.subject_df.sort_values(by="publicationdateyear", inplace=True)
        self.counts_dict = self.getCountsPerYear()

    def getCountsPerYear(self):
        """
        Count the occurrences of each topic per year.

        Returns:
            dict: Dictionary with years as keys and nested dictionaries of topics and their counts as values.
        """
        counts = defaultdict(lambda: defaultdict(int))

        for idx, row in self.subject_df.iterrows():
            year = row["publicationdateyear"]
            subject = row["subject"]

            if pd.isna(year) or pd.isna(subject):
                continue

            subjects = subject.split(", ")
            for sub in subjects:
                counts[year][sub] += 1

        sorted_counts = {
            year: dict(subjects) for year, subjects in sorted(counts.items())
        }

        return sorted_counts

    def getIntervalCounts(self, start_date, end_date):
        """
        Get the counts of topics within a specified interval.

        Args:
            start_date (int): Start year of the interval.
            end_date (int): End year of the interval.

        Returns:
            dict: Dictionary of topics and their counts within the specified interval.
        """
        cts = {}

        for year, year_counts in self.counts_dict.items():
            if start_date <= year <= end_date:
                for topic, cnt in year_counts.items():
                    cts[topic] = cts.get(topic, 0) + cnt

        if not cts:
            print("No counts found in the specified interval.")
            return None

        sorted_cts = dict(sorted(cts.items(), key=lambda x: x[1], reverse=True))

        return sorted_cts

    def plotInterval(self, start_date, end_date, n, top=True):
        """
        Plot the top or least frequent topics within a specified interval.

        Args:
            start_date (int): Start year of the interval.
            end_date (int): End year of the interval.
            n (int): Number of topics to display.
            top (bool): If True, plot the top n topics. If False, plot the least frequent n topics.
        """
        counts = self.getIntervalCounts(start_date, end_date)

        if counts is None:
            return

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=top)

        if top:
            items = sorted_counts[:n]
        else:
            items = sorted_counts[:-1][:n]

        topics, frequencies = zip(*items)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(topics, frequencies, color="skyblue")

        for bar, freq in zip(bars, frequencies):
            plt.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                str(freq),
                ha="left",
                va="center",
                color="black",
            )

        plt.xlabel("Frequency")
        plt.ylabel("Topic")
        plt.title(
            "Top"
            if top
            else "Least" + f" {n} Frequencies in Interval {start_date}-{end_date}"
        )
        plt.gca().invert_yaxis()  # Invert y-axis to display the highest frequency at the top
        plt.show()

    def save(self, filepath):
        """
        Save the FrequencyCounter object to a pickle file.

        Args:
            filepath (str): Path to the pickle file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Load a FrequencyCounter object from a pickle file.

        Args:
            filepath (str): Path to the pickle file.

        Returns:
            FrequencyCounter: Loaded FrequencyCounter object.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/eric_records.csv")
    counter = FrequencyCounter(df=df)
    counter.plotInterval(2000, 2010, 10, True)
