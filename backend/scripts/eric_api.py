import json
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests


class EricApi:
    """
    A class to interact with the ERIC (Education Resources Information Center) API.
    """

    def __init__(self) -> None:
        pass

    def getEricRecords(self, search, fields=None, start=0, rows=200):
        """
        Retrieve ERIC records based on a search query.

        Args:
            search (str): The search query.
            fields (list, optional): List of fields to include in the response. Defaults to None.
            start (int, optional): The start index of the records to retrieve. Defaults to 0.
            rows (int, optional): The number of records to retrieve per request. Defaults to 200.

        Returns:
            pandas.DataFrame: DataFrame containing the retrieved records.
        """
        url = "https://api.ies.ed.gov/eric/?"
        url = (
            url
            + "search="
            + search
            + "&rows="
            + str(rows)
            + "&format=json&start="
            + str(start)
        )
        if fields:
            url = url + "&fields=" + ", ".join(fields)
        responseJson = requests.get(url).json()
        return pd.DataFrame(responseJson)

    def getRecordCount(self, search):
        """
        Get the total number of records for a given search query.

        Args:
            search (str): The search query.

        Returns:
            int: The total number of records.
        """
        dataFrame = self.getEricRecords(search)
        totalRecords = dataFrame.loc["numFound"][0]
        print("Search", search, "returned", "{:,}".format(totalRecords), "records")
        return totalRecords

    def cleanElementsUsingList(self, x):
        """
        Clean elements in a list.

        Args:
            x (list): The list to clean.

        Returns:
            str: Comma-separated string if not empty, otherwise None.
        """
        if not isinstance(x, list):
            return x
        if not x or (len(x) == 1 and x[0] == ""):
            return None
        return ", ".join(x)

    def getAllEricRecords(self, search, fields=None, cleanElements=True):
        """
        Retrieve all ERIC records for a given search query.

        Args:
            search (str): The search query.
            fields (list, optional): List of fields to include in the response. Defaults to None.
            cleanElements (bool, optional): Whether to clean elements in lists. Defaults to True.

        Returns:
            pandas.DataFrame: DataFrame containing all retrieved records.
        """
        startTime = time.time()
        nextFirstRecord = 0
        numRecordsReturnedEachApiCall = 200
        totalRecords = self.getRecordCount(search)
        if totalRecords == 0:
            print("Search", search, "has no results")
            return []

        while nextFirstRecord < totalRecords:
            df = self.getEricRecords(search, fields, nextFirstRecord)
            if nextFirstRecord == 0:
                records = pd.DataFrame(df.loc["docs"][0])
            else:
                records = pd.concat(
                    [records, pd.DataFrame(df.loc["docs"][0])],
                    sort=False,
                    ignore_index=True,
                )
            nextFirstRecord += numRecordsReturnedEachApiCall
        print("took", "{:,.1f}".format(time.time() - startTime), "seconds")
        return (
            records.applymap(self.cleanElementsUsingList) if cleanElements else records
        )