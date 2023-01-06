from tkinter import *

def _command():
    print("hihi")
    print("holy shiet")
root = Tk()

root.geometry('100x100')

btn = Button(root, text = 'Click me', bd = '5', command = _command())

btn.pack(side='top')

root.mainloop()