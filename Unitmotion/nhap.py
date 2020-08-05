import pandas as pd

c1 = pd.read_csv('machine/Calling/Sensors_1.csv')
c2 = pd.read_csv('machine/Calling/Sensors_2.csv')
c3 = pd.read_csv('machine/Calling/Sensors_3.csv')
c4 = pd.read_csv('machine/Calling/Sensors_4.csv')
c5 = pd.read_csv('machine/Calling/Sensors_5.csv')
c6 = pd.read_csv('machine/Calling/Sensors_6.csv')
c7 = pd.read_csv('machine/Calling/Sensors_7.csv')
c8 = pd.read_csv('machine/Calling/Sensors_8.csv')
c9 = pd.read_csv('machine/Calling/Sensors_9.csv')
c10 = pd.read_csv('machine/Calling/Sensors_10.csv')
calling = pd.concat([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], axis = 0)


t1 = pd.read_csv('machine/Texting/Sensors_1.csv')
t2 = pd.read_csv('machine/Texting/Sensors_2.csv')
t3 = pd.read_csv('machine/Texting/Sensors_3.csv')
t4 = pd.read_csv('machine/Texting/Sensors_4.csv')
t5 = pd.read_csv('machine/Texting/Sensors_5.csv')
t6 = pd.read_csv('machine/Texting/Sensors_6.csv')
t7 = pd.read_csv('machine/Texting/Sensors_7.csv')
t8 = pd.read_csv('machine/Texting/Sensors_8.csv')
t9 = pd.read_csv('machine/Texting/Sensors_9.csv')
t10 = pd.read_csv('machine/Texting/Sensors_10.csv')
texting = pd.concat([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10], axis = 0)


s1 = pd.read_csv('machine/Swinging/Sensors_1.csv')
s2 = pd.read_csv('machine/Swinging/Sensors_2.csv')
s3 = pd.read_csv('machine/Swinging/Sensors_3.csv')
s4 = pd.read_csv('machine/Swinging/Sensors_4.csv')
s5 = pd.read_csv('machine/Swinging/Sensors_5.csv')
s6 = pd.read_csv('machine/Swinging/Sensors_6.csv')
s7 = pd.read_csv('machine/Swinging/Sensors_7.csv')
s8 = pd.read_csv('machine/Swinging/Sensors_8.csv')
s9 = pd.read_csv('machine/Swinging/Sensors_9.csv')
s10 = pd.read_csv('machine/Swinging/Sensors_10.csv')
swinging = pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10], axis = 0)


p1 = pd.read_csv('machine/Pocket/Sensors_1.csv')
p2 = pd.read_csv('machine/Pocket/Sensors_2.csv')
p3 = pd.read_csv('machine/Pocket/Sensors_3.csv')
p4 = pd.read_csv('machine/Pocket/Sensors_4.csv')
p5 = pd.read_csv('machine/Pocket/Sensors_5.csv')
p6 = pd.read_csv('machine/Pocket/Sensors_6.csv')
p7 = pd.read_csv('machine/Pocket/Sensors_7.csv')
p8 = pd.read_csv('machine/Pocket/Sensors_8.csv')
p9 = pd.read_csv('machine/Pocket/Sensors_9.csv')
p10 = pd.read_csv('machine/Pocket/Sensors_10.csv')
pocket = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10], axis = 0)


total = pd.concat([calling,texting,swinging,pocket], axis = 0)
total.to_csv('machine/total.csv', index=False)

a = total.loc[total['Activities'] == 2]
a.to_csv('machine/Dynamic.csv', index=False)

b = total.loc[total['Activities'] == 1]
b.to_csv('machine/Static.csv', index=False)

a = pd.read_csv('machine/Dynamic.csv')
c = a.query('GroupType != 1 and GroupType !=2 and GroupType !=3 ')
c.to_csv('machine/HoldingStyle.csv', index=False)
#print(c)

print('Done')