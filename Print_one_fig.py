import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk


import pathlib

def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    
    
def show_image(file_dir):
    
    data_dir_train = file_dir
    data_dir_train = pathlib.Path(data_dir_train)
    angular_leaf_spot = list(data_dir_train.glob('angular_leaf_spot/*'))
    image = ImageTk.PhotoImage(file=angular_leaf_spot[0])
    imagebox.config(image=image)
    imagebox.image = image  
#    
    
    
root = tk.Tk()
folder_path = tk.StringVar()
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, text="QUIT", fg="red", command=lambda: root.destroy())
button.pack(side=tk.LEFT)

slogan = tk.Button(frame, text="Browse", command=browse_button)
slogan.pack(side=tk.LEFT)

other = tk.Button(frame, text="Compute", command=lambda: show_image(folder_path.get()))
other.pack(side=tk.LEFT)

# label to show the image
imagebox = tk.Label(root)
imagebox.pack()

root.mainloop()