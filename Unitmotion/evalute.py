# import matplotlib.pyplot as plt
# import pandas as pd
#
# eva = pd.read_csv('Evaluation.csv')
#
#
# staticdynamic = eva.loc[(eva['Name'] == 'Static Dynamic') & (eva['ML'] != 'ETC10') & (eva['ML'] != 'ETC50')
#                 & (eva['ML'] != 'ETC25') ]

#
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# labels = ['G1', 'G2', 'G3', 'G4']
# men_means = [20, 34, 30, 35]
# women_means = [25, 32, 34, 20]
# kid = [10,21,15,25]
# pon = [9,8,13,20]
#
#
# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')
# rects3 = ax.bar(x , kid, width, label='kid')
# rects4 = ax.bar(x , pon, width, label='pon')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# fig.tight_layout()
#
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# set width of bar
# barWidth = 0.2

# set height of bar



#
# # Set position of bar on X axis
# r1 = np.arange(len(DT))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]
#
# # Make the plot
# plt.bar(r1, DT, color='#7f6d5f', width=barWidth, edgecolor='white', label='DT')
# plt.bar(r2, KNN3, color='#557f2d', width=barWidth, edgecolor='white', label='KNN3')
# plt.bar(r3, KNN5, color='#2d7f5e', width=barWidth, edgecolor='white', label='KNN5')
# plt.bar(r4, ETC, color='b', width=barWidth, edgecolor='white', label='ETC')
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Accuracy of Static & Dynamic', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(DT))], ['802', '502', '601', '401'])
#
# # Create legend & Show graphic
# plt.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


if __name__ == '__main__':
    eva = pd.read_csv('machine/Evaluation317.csv')
    titlearr = ['Total','Holding Styles','Non-Moving','Moving','Phone']
    for va in range(len(titlearr)):
        title = titlearr[va]

        DT = []
        KNN3 = []
        KNN7 = []
        ETC = []
        RFC = []
        featurename = 'Max'
        #a = eva.loc[(eva['Name'] == f'{title}') & eva['ML'].isin(['KNN3','KNN5','DT','RFC100','ETC100'])]
        a = eva.loc[(eva['Name'] == f'{title}') & (eva['Feature'] == f'{featurename}') & eva['ML'].apply(lambda x: x in ['KNN3','KNN7','DT','RFC100','ETC100'])]
        a = a.reset_index(drop=True)
        #('802', '502', '601', '401')
        predictname = 'Accuracy'
        for i in range(len(a.Name)):
            if a['ML'][i] == 'DT':
                DT.append(a[f'{predictname}'][i])
            elif a['ML'][i] == 'KNN3':
                KNN3.append(a[f'{predictname}'][i])
            elif a['ML'][i] == 'KNN7':
                KNN7.append(a[f'{predictname}'][i])
            elif a['ML'][i] == 'ETC100':
                ETC.append(a[f'{predictname}'][i])
            else:
                RFC.append(a[f'{predictname}'][i])

        # DT = [0.80,0.79,0.84,0.82]
        # KNN3 = [0.93,0.88,0.92,0.90]
        # KNN5 = [0.92,0.87,0.92,0.89]
        # ETC = [0.97,0.95,0.97,0.96]
        # RFC = [0.65,0.80,0.90,0.79]
        #print(a.sort_values(by='Type'))


        N = 4
        ind = np.arange(N)  # the x locations for the groups
        width = 0.15      # the width of the bars

        fig = plt.figure()
        ax = fig.add_subplot(111)

        rects1 = ax.bar(ind - width*2, KNN3, width, color='#AD4F0E')
        rects2 = ax.bar(ind - width, KNN7, width, color='#2d7f5e')
        rects3 = ax.bar(ind, DT, width, color='#11E8D4')
        rects4 = ax.bar(ind+width, RFC, width, color='#557f2d')
        rects5 = ax.bar(ind+width*2, ETC, width, color='#1138E8')

        plt.xlabel(f'{predictname} of {title} ({featurename})', fontweight='bold')

        ax.set_ylabel(f'{predictname} (%)')
        ax.set_xticks(ind)
        ax.set_xticklabels( ('Overlap 80% - Windowtime 2s', 'Overlap 50% - Windowtime 2s', 'Overlap 60% - Windowtime 1s', 'Overlap 40% - Windowtime 1s') )
        ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('KNN3', 'KNN7', 'DT','RFC','ETC'),loc='lower right')

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        autolabel(rects5)
        plt.show()


