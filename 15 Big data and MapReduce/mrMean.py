# Mrjob implementation of distributed mean variance calculation
from mrjob.job import MRJob
from mrjob.step import MRStep


class MRmean(MRJob):
    def __init__(self, *args, **kwargs):
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    # Receives streaming inputs
    def map(self, key, val):
        if False:
            yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal * inVal

    # Processing after all inputs have arrived
    def map_final(self):
        mn = self.inSum / self.inCount
        mnSq = self.inSqSum / self.inCount
        yield (1, [self.inCount, mn, mnSq])

    def reduce(self, key, packedValues):
        cumVal, cumSumSq, cumN = 0.0, 0.0, 0.0
        for valArr in packedValues:
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj * float(valArr[1])
            cumSumSq += nj * float(valArr[2])
        mean = cumVal / cumN
        var = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean) / cumN
        yield (mean, var)

    def steps(self):
        return ([MRStep(mapper=self.map, mapper_final=self.map_final,
                        reducer=self.reduce)])


if __name__ == '__main__':
    MRmean.run()
