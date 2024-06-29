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

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import pathlib
import matplotlib.pyplot as plt

plt.ioff()
def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    
    
def show_image(file_dir):
    
    data_dir_train = file_dir
    data_dir_train = pathlib.Path(data_dir_train)
    
    batch_size = 32
    img_height = 180
    img_width = 180
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir_train,
      validation_split=0.8,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir_train,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    class_names = train_ds.class_names
    
    fig = plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        
    canvas = FigureCanvasTkAgg(fig,
                               master = root)  
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   root)
    toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()
    plt.close()
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


root.mainloop()