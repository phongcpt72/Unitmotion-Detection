import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('machine/Evaluation317.csv')

    a = df.loc[(df['Name'].isin(['Phone'])) & (df['Feature'] == 'Max') & (df['Type'] == 802) & df['ML'].apply(
        lambda x: x in['DT','RFC10','ETC10'])]

    a.to_csv('machine/Result.csv', index = False)


    #
    # a = df.loc[(df['Name'] == 'Holding Styles') & (df['Feature'] == 'Max') & (df['Type'] == 802) & df['ML'].apply(
    #     lambda x: x not in ['DT', 'KNN3', 'KNN7'])]
    # a.to_csv('machine/Holding-StylesResult.csv', index=False)
    #
    # a = df.loc[(df['Name'] == 'Moving') & (df['Feature'] == 'Max') & (df['Type'] == 802) & df['ML'].apply(
    #     lambda x: x not in ['DT', 'KNN3', 'KNN7'])]
    # a.to_csv('machine/MovingResult.csv', index=False)
    #
    # a = df.loc[(df['Name'] == 'Non-Moving') & (df['Feature'] == 'Max') & (df['Type'] == 802) & df['ML'].apply(
    #     lambda x: x not in ['DT', 'KNN3', 'KNN7'])]
    # a.to_csv('machine/Non-MovingResult.csv', index=False)
    #
    # a = df.loc[(df['Name'] == 'Phone') & (df['Feature'] == 'Max') & (df['Type'] == 802) & df['ML'].apply(
    #     lambda x: x not in ['DT', 'KNN3', 'KNN7'])]
    # a.to_csv('machine/PhoneResult.csv', index=False)


