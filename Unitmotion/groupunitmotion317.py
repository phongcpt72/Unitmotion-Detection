import pandas as pd

if __name__ == '__main__':
    olarr = ['overlap60_1s','overlap40_1s','overlap50_2s','overlap80_2s']
    for i in range(len(olarr)):
        namestring = olarr[i]
        filename = 'total'
        df = pd.read_csv(f'machine/{namestring}/ML/{filename}.csv')
        grouptype = df['GroupType']
        NewGroup = []
        for i in range(len(grouptype)):
            a = grouptype[i]
            if a == 0 or a == 1:
                NewGroup.append('Non-Moving')
            elif a == 2 or a == 3:
                NewGroup.append('Phone')
            else:
                NewGroup.append('Moving')


        df['NewGroup'] = NewGroup

        #df.to_csv(f'machine/{namestring}/newgroup317/{filename}.csv', index= False)
        # a = df.loc[(df['NewGroup'] == 'Non-Moving')]
        # a.to_csv(f'machine/{namestring}/newgroup317/Non-Moving.csv', index= False)
        #
        # b = df.loc[(df['NewGroup'] == 'Phone')]
        # b.to_csv(f'machine/{namestring}/newgroup317/Phone.csv', index=False)
        #
        # c = df.loc[(df['NewGroup'] == 'Moving')]
        # c.to_csv(f'machine/{namestring}/newgroup317/Moving.csv', index=False)

        d = df.loc[df['GroupType'].apply(lambda x: x not in [1,2,3])]
        d.to_csv(f'machine/{namestring}/newgroup317/HoldingStyles.csv', index=False)

        print("Done")