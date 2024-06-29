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
    global model_name
    global class_names
    data_dir_train = file_dir
    data_dir_train = pathlib.Path(data_dir_train)
    
    batch_size = 32
    img_height = 180
    img_width = 180
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir_train,
      validation_split=0.2,
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
    
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    
    
    
    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                      img_width,
                                      3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )
      
      
    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])
      
      
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    epochs = 15
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    #plt.show()
        
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
    model_name = model


#model = show_image(folder_path.get())


def Check():
    flower_url = text_entry.get()
    
    flower_path = tf.keras.utils.get_file('', origin=flower_url)
    
    img = tf.keras.utils.load_img(
    flower_path, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model_name.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    res_txt["text"] = "This image most likely belongs to "+str(class_names[np.argmax(score)])+" with a "+str(round(100 * np.max(score),2))+" percent confidence."

    
    
root = tk.Tk()
folder_path = tk.StringVar()
model_name = None
class_names = None
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, text="QUIT", fg="red", command=lambda: root.destroy())
button.pack(side=tk.LEFT)

Browse = tk.Button(frame, text="Browse", command=browse_button)
Browse.pack(side=tk.LEFT)

Compute = tk.Button(frame, text="Compute", command=lambda:show_image(folder_path.get()))
Compute.pack(side=tk.LEFT)

text_entry =  tk.Entry(master = root, width = 50)
text_entry.pack(side=tk.LEFT)


Check_file = tk.Button(frame, text="Check", command = Check)
Check_file.pack()
# label to show the image

res_txt = tk.Label(master= frame)
res_txt.pack()
root.mainloop()