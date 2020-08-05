class banbe:
    ten = 'khong ten'
    tuoi = '12'
    def __init__(self, ten, tuoi):
        self.ten = ten
        self.tuoi = tuoi
    def chao(self):
        print('Xin chao, toi ten la', self.ten, self.tuoi)


banbe1 = banbe('Phong','12')
banbe2 = banbe('VAnh','22')
banbe3 = banbe('Cun','31')
nhom = [banbe1, banbe2,banbe3]
print(nhom)
for ban in nhom:
    ban.chao()