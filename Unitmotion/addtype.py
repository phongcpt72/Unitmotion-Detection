

import pandas as pd

def changetolist(arr):
    tmp = []
    for i in range(len(arr)):
        tmp.append(arr[i])
    return tmp
if __name__ == '__main__':

    posingname= ['Calling','Texting','Swinging','Pocket']
    for i in range(len(posingname)):
        namepose = posingname[i]

        filessss = 1
        while filessss < 11:
            print(filessss)
            file = pd.read_csv(f'CleanData/{namepose}/Sensors_{filessss}.csv')
            savename = f'addtype/{namepose}/Sensors_{filessss}.csv'
            filessss += 1
            ax = file.Ax
            ay = file.Ay
            az = file.Az
            gx = file.Gx
            gy = file.Gy
            gz = file.Gz
            mx = file.Mx
            my = file.My
            mz = file.Mz
            liax = file.LiAx
            liay = file.LiAy
            liaz = file.LiAz
            p = file.P
            value = file.Value
            type = file.Type


            idtmp = []
            sums = 0
            for i in range(len(ax)):
                idtmp.append(sums)
                sums = round(sums+0.04,2)

            count = 1
            for i in range(len(ax)):
                if value[i] == 0:
                    type[i] = 0
                elif value[i] == 1:
                    type[i] = count
                    if value[i+1] == 0 and i <= len(ax):
                        count+=1

            i = 0
            while True:
                if type[i] == 0:
                    type[i] = 13
                    i+=1
                else:
                    break
            tmptype = []
            for i in range(len(ax)):
                if type[i] == 0: # standing
                    tmptype.append(0)
                elif type[i] == 13:
                    tmptype.append(1)
                elif type[i] == 1 : # grab the phone
                    tmptype.append(2)
                elif type[i] == 11: #put the phone
                    tmptype.append(3)
                elif type[i] == 3 or type[i] == 9: # pass the door
                    tmptype.append(5)
                elif type[i]%2 == 0: # walking
                    tmptype.append(4)
                elif type[i] == 5: # downstair
                    tmptype.append(6)
                elif type[i] == 7: # upstair
                    tmptype.append(7)


            eachtype = changetolist(type)
            i = -1
            while True:
                if eachtype[i] == 0:
                    eachtype[i] = 13
                    tmptype[i] = 1
                    i -= 1
                else:
                    break

            activities = []

            for i in range(len(eachtype)):
                if tmptype[i] == 0 or tmptype[i] == 1:
                    activities.append(1)
                else:
                    activities.append(2)


            save = pd.DataFrame({"Timestamp": idtmp,"Ax":ax,"Ay":ay,"Az":az,"Gx":gx,"Gy":gy,"Gz":gz,"Mx":mx,"My":my,"Mz":mz,
                                 "LiAx": liax, "LiAy": liay, "LiAz": liaz,"P": p,"Value":value,"Type":eachtype,"GroupType": tmptype,
                                 "Activities": activities})

            save.to_csv(savename, index = False)
        print('Done')
