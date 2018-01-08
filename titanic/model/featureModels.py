import copy as _

class featureModel:
    def __init__(self, trainDataset, submissionDataset, excludes):
        self.featureExcludes = excludes
        self.trainDataset = _.deepcopy(trainDataset)
        self.submissionDataset = _.deepcopy(submissionDataset)

    def getDatasets(self):
        return {
            'train' : self.trainDataset.drop(self.featureExcludes, axis=1),
            'submission' : self.submissionDataset.drop(self.featureExcludes, axis = 1)
        }