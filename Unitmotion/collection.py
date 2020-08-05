col2 = [
                'IqrMagAcc',
                'MeanMagAcc',
                'STDMagAcc',
                'STDMagGyr',
                'STDMagMag', 'STDMagLiAcc',
                'VarMagAcc',
                'VarMagGyr',
                'VarMagMag',
                'VarMagLiAcc'
]

feature_cols = [
                'MaxAx','MaxAy','MaxAz',
                'MaxMx','MaxMy','MaxMz',
                'MaxLiAx','MaxLiAy','MaxLiAz',

                'MinAx','MinAy','MinAz',
                'MinLiAx','MinLiAy','MinLiAz',

                'MeanMx', 'MeanMy', 'MeanMz', 'MeanLiAx', 'MeanLiAy', 'MeanLiAz',
                'MeanMx','MeanMy','MeanMz',
                'MeanLiAx','MeanLiAy','MeanLiAz',

                'MedianAx', 'MedianAy', 'MedianAz',
                'MedianMx', 'MedianMy', 'MedianMz',

                'StandardDeviationAx','StandardDeviationAy','StandardDeviationAz',
                'StandardDeviationGx','StandardDeviationGy','StandardDeviationGz',
                'StandardDeviationMx','StandardDeviationMy','StandardDeviationMz',
                'StandardDeviationLiAx','StandardDeviationLiAy','StandardDeviationLiAz',

                'VarianceAx', 'VarianceAy', 'VarianceAz',
                'VarianceGx', 'VarianceGy', 'VarianceGz',
                'VarianceLiAx','VarianceLiAy','VarianceLiAz',

                'SMAAcc',
                'SMAMag',
                'SMALiAcc',

                ]
from collections import Counter

a = dict(Counter(feature_cols))

sums = 0
for x in a.values():
    if x >= 2:
        sums+=2
    else:
        sums+=x

print(a)
print(sums)