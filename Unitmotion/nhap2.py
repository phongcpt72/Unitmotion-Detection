import pandas as pd

filecur = 'overlap80_2s'
over1 = pd.read_csv(f'machine/{filecur}/newgroup317/totalml.csv')
over2 = pd.read_csv(f'machine/{filecur}/newgroup317/Movingml.csv')
over3 = pd.read_csv(f'machine/{filecur}/newgroup317/Non-Movingml.csv')
over4 = pd.read_csv(f'machine/{filecur}/newgroup317/Phoneml.csv')
over5 = pd.read_csv(f'machine/{filecur}/newgroup317/HoldingStylesml.csv')
over802 = pd.concat([over1,over2,over3,over4,over5], axis= 0)


filecur = 'overlap50_2s'
over1 = pd.read_csv(f'machine/{filecur}/newgroup317/totalml.csv')
over2 = pd.read_csv(f'machine/{filecur}/newgroup317/Movingml.csv')
over3 = pd.read_csv(f'machine/{filecur}/newgroup317/Non-Movingml.csv')
over4 = pd.read_csv(f'machine/{filecur}/newgroup317/Phoneml.csv')
over5 = pd.read_csv(f'machine/{filecur}/newgroup317/HoldingStylesml.csv')
over502 = pd.concat([over1,over2,over3,over4,over5], axis= 0)


filecur = 'overlap60_1s'
over1 = pd.read_csv(f'machine/{filecur}/newgroup317/totalml.csv')
over2 = pd.read_csv(f'machine/{filecur}/newgroup317/Movingml.csv')
over3 = pd.read_csv(f'machine/{filecur}/newgroup317/Non-Movingml.csv')
over4 = pd.read_csv(f'machine/{filecur}/newgroup317/Phoneml.csv')
over5 = pd.read_csv(f'machine/{filecur}/newgroup317/HoldingStylesml.csv')
over601 = pd.concat([over1,over2,over3,over4,over5], axis= 0)


filecur = 'overlap40_1s'
over1 = pd.read_csv(f'machine/{filecur}/newgroup317/totalml.csv')
over2 = pd.read_csv(f'machine/{filecur}/newgroup317/Movingml.csv')
over3 = pd.read_csv(f'machine/{filecur}/newgroup317/Non-Movingml.csv')
over4 = pd.read_csv(f'machine/{filecur}/newgroup317/Phoneml.csv')
over5 = pd.read_csv(f'machine/{filecur}/newgroup317/HoldingStylesml.csv')
over401 = pd.concat([over1,over2,over3,over4,over5], axis= 0)

total = pd.concat([over802,over502,over601,over401], axis = 0)
total.to_csv('machine/Evaluation317.csv', index=False)

print('Done')

#
# over1 = pd.read_csv(f'machine/{filecur}/ML/totalml.csv')
# over2 = pd.read_csv(f'machine/{filecur}/ML/StaticDynamicml.csv')
# over3 = pd.read_csv(f'machine/{filecur}/ML/Staticml.csv')
# over4 = pd.read_csv(f'machine/{filecur}/ML/Dynamicml.csv')
# over5 = pd.read_csv(f'machine/{filecur}/ML/HoldingStyleml.csv')