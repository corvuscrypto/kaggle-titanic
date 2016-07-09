from modeling import *
import numpy as np
import random


class TrainingRepo(object):
    def __init__(self, csv_file):

        if isinstance(csf_file):
            csv_file = open(csv_file, 'r')
        if not isinstance(csv_file, file):
            raise TypeError("csv_file must be a file object or a string pointing to the file path")

        reader = csv.reader(csv_file)

        # this will be an array of tuples
        self.data = []

        # load in the data
        for row in reader:
            # assume the result is in second column for train data
            self.data.append( (data2vec(row), int(row[1])) )

        csv_file.close()

    def get_batch(self, size=20):

        # always reshuffle when getting a new batch
        random.shuffle(self.data)

        features = []
        results = []
        # grab some data
        for _ in range(size):
            row = self.data.pop()
            features.append(row[0])
            results.append(row[1])

        #transform into numpy matrix/vector for features/result
        features = np.matrix(features)
        results = np.array(results)

        return features, results

class TestingRepo(object):
    def __init__(self, feature_file, results_file):

        if isinstance(feature_file):
            feature_file = open(feature_file, 'r')
        if not isinstance(feature_file, file):
            raise TypeError("feature_file must be a file object or a string pointing to the file path")

        if isinstance(results_file):
            results_file = open(results_file, 'r')
        if not isinstance(results_file, file):
            raise TypeError("results_file must be a file object or a string pointing to the file path")

        freader = csv.reader(feature_file)
        rreader = csv.reader(results_file)

        # take the lazy approach for this one and assume that the data organization in results and feature files
        # are 1-1 joined by id

        # this will be an array of tuples
        self.data = []
        i=0
        # load in the data
        for row in freader:
            result = int(rreader[i][1])
            # pad the row to ensure proper conditioning
            row.insert(1,0)
            self.data.append( (data2vec(row), result) )

        feature_file.close()
        results_file.close()

    def get_batch(self, size=20):

        # always reshuffle when getting a new batch
        random.shuffle(self.data)

        features = []
        results = []
        # grab some data
        for _ in range(size):
            row = self.data.pop()
            features.append(row[0])
            results.append(row[1])

        #transform into numpy matrix/vector for features/result
        features = np.matrix(features)
        results = np.array(results)

        return features, results
