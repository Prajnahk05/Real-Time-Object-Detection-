from tkinter import*
from PIL import Image,ImageTk
import os


win=Tk()
win.title("Object Detection Using Python")
win.geometry("1300x1000")



img=Image.open("pic2.jpeg")
bim=ImageTk.PhotoImage(img)
l1=Label(win,image=bim)
l1.pack()
def objectdetection():
    os.system("python detectobject1.py")

l = Label(win, text="Object Detection Using Python", font=("algerian", 30, "bold", "italic"), fg="black", bg="white")
l.place(x=300, y=20)

b2 = Button(win, text="DETECTION", bg="light blue", command=objectdetection, fg="navy blue", font=("times", 16, "bold", "italic"), padx="20px",
            pady="15px")
b2.place(x=550, y=285)


win.mainloop()
