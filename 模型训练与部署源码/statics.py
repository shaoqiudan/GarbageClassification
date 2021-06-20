# coding=utf-8
import os
import xlwt
x = []
y = []
for dirpath, dirnames, filenames in os.walk("E:\\ruanjian\\demo\\pachong\\deepleaning\\垃圾图片库"):
    file_count = 0
    for file in filenames:
        file_count = file_count + 1

    x.append(dirpath)
    y.append(file_count)
workbook = xlwt.Workbook(encoding='utf-8')
data_sheet = workbook.add_sheet("show_data")
row = 0
col = 0

for row in range(0, len(x)):
    data_sheet.write(row, 0, x[row])
    data_sheet.write(row, 1, y[row])

xls_path = os.path.join("E:/", "Excel.xls")
workbook.save(xls_path)


