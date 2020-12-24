from socket import *

HOST = '172.18.167.10'
PORT = 21567
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET, SOCK_STREAM)
tcpCliSock.connect(ADDR)

import io
from tkinter import *
import tkinter as tk
from urllib.request import urlopen
from PIL import Image, ImageTk

root = tk.Tk()
path = "./0.5.png"
pil_image = Image.open(path)
w, h = pil_image.size
#fname = path.split('/')[-1]
fname = 'lena.png'
sf = "{} ({}x{})".format(fname, w, h)
root.title(sf)

tk_image = ImageTk.PhotoImage(pil_image)
img_label = tk.Label(root, image=tk_image, bg='white')
img_label.pack(padx=0, pady=0)


sel = tk.Label(root, bg='green', fg='white', text='Done!', width=50)
sel.pack()

'''
def printAndShow(i):
    sel.config(text='you have selected alpha ' + i)
    png =ImageTk.PhotoImage(Image.open('./low.png'))
    img_label.config(image=png)
    img_label.image=png
'''
v=StringVar()

s = tk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, length=400, showvalue=1,tickinterval=0.1, resolution=0.001, variable=v)
s.set(0.5)
s.pack()

def change():
    message = v.get()
    print(message)
    sel.config(text = "Generating image with the given alpha value "+ message +"......")
    root.update()
    tcpCliSock.send(bytes(message, 'utf-8'))
    data = tcpCliSock.recv(BUFSIZ)

    if data.decode() == "0001":
        print("Sorr file %s not found"%message)
    else:
        tcpCliSock.send("File size received".encode())
        file_total_size = int(data.decode())
        received_size = 0
        f = open("new.png" ,"wb")
        while received_size < file_total_size:
            data = tcpCliSock.recv(BUFSIZ)
            f.write(data)
            received_size += len(data)
            print("已接收:",received_size)
        f.close()
        print("receive done",file_total_size," ",received_size)


    png =ImageTk.PhotoImage(Image.open('./new.png'))
    img_label.config(image=png)
    img_label.image=png


    sel.config(text = "Done!")

b = tk.Button(root, text="Update!", command=change)
b.pack()

root.mainloop()