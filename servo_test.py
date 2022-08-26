import serial
import tkinter as tk

#%%

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 300)
canvas1.pack()

entry1 = tk.Entry (root) 
entry2 = tk.Entry (root) 
canvas1.create_window(200, 100, window=entry1)
canvas1.create_window(200, 140, window=entry2)

arduino = serial.Serial('COM3', 115200, timeout=5)
print("arduino connected") # connect to arduino

def servo():
    x = entry1.get()
    y = entry2.get()
    inf = str(x) + ' ' + str(y) + ';'
    arduino.write(bytes(inf, 'utf-8'))
    
button1 = tk.Button(text='Servos move', command=servo)
canvas1.create_window(200, 180, window=button1)

root.mainloop()