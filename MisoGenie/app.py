import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk,Image
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from matplotlib import pyplot as plt

import datetime


def select_image(still_image_frame,lb2,label_yes):
    global img_path
    global emotion_prompt
    img_path = tk.filedialog.askopenfilename()
    lb2 = ctk.CTkLabel(still_image_frame,text="Image selected {}".format(img_path), fg_color='#e8e4da',text_color='white',corner_radius= 10)
    lb2.pack()

    img = cv2.imread(img_path)
    model = keras.models.load_model('action_unit_detection_v5')

    au_mapping = {
        0: 'Inner Brow Raiser',
        1: 'Outer Brow Raiser',
        2: 'Neutral',
        3: 'Brow Lowerer',
        4: 'Upper Lid Raiser',
        5: 'Cheek Raiser',
        6: 'Lid Tightener',
        7: 'Neutral',
        8: 'Nose Wrinkler',
        9: 'Upper Lip Raiser',
        10: 'Nasolabial Deepener',
        11: 'Lip Corner Puller',
        12: 'Cheek Puffer',
        13: 'Dimpler',
        14: 'Lip Corner Depressor',
        15: 'Lower Lip Depressor',
        16: 'Chin Raiser',
        17: 'Lip Puckerer',
        18: 'Neutral',
        19: 'Lip Puckerer',
        20: 'Neutral',
        21: 'Lip Funneler',
        22: 'Lip Tightener',
        23: 'Lip Pressor',
        24: 'Lips Part',
        25: 'Jaw Drop',
        26: 'Mouth Stretch',
        27: 'Lip Suck',
    }
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces = detector.detectMultiScale(img1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    emotion = ''
    for (x, y, w, h) in faces:
        face = img1[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        action_units = model.predict(face)[0]

        predicted_facs_au = [au_mapping[i] for i, value in enumerate(action_units) if value >= 0.5]
        if 'Lips Part' and 'Lip Corner Puller' in predicted_facs_au:
            emotion = 'Happy'
        elif 'Brow Lowerer' and  'Lip Corner Depressor' in predicted_facs_au:
            emotion = 'Sadness'
        elif 'Lips Part' and  'Mouth Stretch' in predicted_facs_au:
            emotion = 'Suprise'
        elif 'Inner Brow Raiser' and 'Brow Lowerer' and 'Lip Stretcher' and 'Lips Part' in predicted_facs_au:
            emotion = 'Fear'
        elif 'Nose Wrinkler' and 'Upper Lip Raiser' and 'Chin Raiser' in predicted_facs_au:
            emotion = 'Disgust'
        elif 'Lid Tightener' and 'Lip Pressor' in predicted_facs_au:
            emotion = 'Anger'
        else:
            emotion = 'Neutral'
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img1, ''.join(emotion), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2)
    if label_yes:
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        desired_size = (400, 400)
        img = Image.fromarray(img)
        img = img.resize(desired_size, Image.ANTIALIAS)
        label = tk.Label(still_image_frame)
        label.image = ImageTk.PhotoImage(img)
        label.configure(image=label.image)
        label.pack()
    else:
        emotion_prompt = "Create a image that can reflect what some person displaying the emotion: " + emotion + " and the facial action units of {}".format(predicted_facs_au) + " will be feeling"
        print(emotion_prompt)

img_path = None
emotion_prompt = None
#Create the app
app = ctk.CTk()

def home_page():
    home_frame = tk.Frame(main_frame)

    lb = ctk.CTkLabel(home_frame, text='Home Page', fg_color='#e8e4da',text_color='white',corner_radius= 10)
    lb.pack()

    image = Image.open('README.png')

    tk_image = ImageTk.PhotoImage(image)

    lb2 = ctk.CTkLabel(home_frame, image=tk_image)
    lb2.pack()

    home_frame.pack()

def still_image_page():
    still_image_frame = tk.Frame(main_frame)

    lb = ctk.CTkLabel(still_image_frame, text='Emotion Detection From Image', fg_color='#e8e4da',text_color='white',corner_radius= 10)
    lb.pack()
    
    select = ctk.CTkButton(still_image_frame,text='Choose Image', fg_color='#e8e4da',text_color='white', command= lambda: select_image(still_image_frame,lb2,True))
    lb2 = ctk.CTkLabel(still_image_frame,text="Image selected {}".format(img_path), fg_color='#e8e4da',text_color='white',corner_radius= 10)
    select.pack()
        
    still_image_frame.update()
    still_image_frame.pack()

def real_time_page():
    real__time_frame = tk.Frame(main_frame)

    lb = ctk.CTkLabel(real__time_frame, text='Emotion Detection Live', fg_color='#e8e4da',text_color='black',font=('Cambria',30), corner_radius= 10)
    lb.pack()
    
    L1 = tk.Label(real__time_frame, bg='red')
    L1.pack()

    snapshot = ctk.CTkButton(real__time_frame, text='Take Snapshot',fg_color='#e8e4da',text_color='black',font=('Cambria',30),command= lambda: Photolelo()).pack()

    real__time_frame.pack()

    model = keras.models.load_model('action_unit_detection_v5')
    # Define a dictionary mapping action unit indexes to FACS AUs
    au_mapping = {
        0: 'Inner Brow Raiser',
        1: 'Outer Brow Raiser',
        2: 'Neutral',
        3: 'Brow Lowerer',
        4: 'Upper Lid Raiser',
        5: 'Cheek Raiser',
        6: 'Lid Tightener',
        7: 'Neutral',
        8: 'Nose Wrinkler',
        9: 'Upper Lip Raiser',
        10: 'Nasolabial Deepener',
        11: 'Lip Corner Puller',
        12: 'Cheek Puffer',
        13: 'Dimpler',
        14: 'Lip Corner Depressor',
        15: 'Lower Lip Depressor',
        16: 'Chin Raiser',
        17: 'Lip Puckerer',
        18: 'Neutral',
        19: 'Lip Puckerer',
        20: 'Neutral',
        21: 'Lip Funneler',
        22: 'Lip Tightener',
        23: 'Lip Pressor',
        24: 'Lips Part',
        25: 'Jaw Drop',
        26: 'Mouth Stretch',
        27: 'Lip Suck',
    }

    # Define the facial landmark detector
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Define the video capture device
    cap = cv2.VideoCapture(0)
    def Photolelo():
        image = Image.fromarray(img1)
        time = str(datetime.datetime.now().today()).replace(":","")+".jpg"
        image.save(time)

    while True:
        img = cap.read()[1]
        img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(img1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        emotion = ''
        for (x, y, w, h) in faces:
            face = img1[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)

            action_units = model.predict(face)[0]

            predicted_facs_au = [au_mapping[i] for i, value in enumerate(action_units) if value >= 0.5]
            if 'Lips Part' and 'Lip Corner Puller' in predicted_facs_au:
                emotion = 'Happy'
            elif 'Brow Lowerer' and  'Lip Corner Depressor' in predicted_facs_au:
                emotion = 'Sadness'
            elif 'Lips Part' and  'Mouth Stretch' in predicted_facs_au:
                emotion = 'Suprise'
            elif 'Inner Brow Raiser' and 'Brow Lowerer' and 'Lip Stretcher' and 'Lips Part' in predicted_facs_au:
                emotion = 'Fear'
            elif 'Nose Wrinkler' and 'Upper Lip Raiser' and 'Chin Raiser' in predicted_facs_au:
                emotion = 'Disgust'
            elif 'Lid Tightener' and 'Lip Pressor' in predicted_facs_au:
                emotion = 'Anger'
            else:
                emotion = 'Neutral'
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img1, ''.join(emotion), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        img = ImageTk.PhotoImage(Image.fromarray(img1))
        L1.configure(image=img)
        real__time_frame.update()


def art_page():
    global emotion_prompt
    art_frame = tk.Frame(main_frame)

    lb = ctk.CTkLabel(art_frame, text='AI art using stable diffusion', fg_color='#e8e4da',text_color='white',corner_radius= 10)
    lb.pack()
    
    select = ctk.CTkButton(art_frame,text='Choose Image', fg_color='#e8e4da',text_color='white', command= lambda: select_image(art_frame,lb2,False))
    lb2 = ctk.CTkLabel(art_frame,text="Image selected {}".format(img_path), fg_color='#e8e4da',text_color='white',corner_radius= 10)
    select.pack()

    lmain = ctk.CTkLabel(art_frame, height=512, width=512)
    lmain.pack()

    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=auth_token)
    pipe = pipe.to(device)

    def generate(): 
        with autocast(device): 
            image = pipe(emotion_prompt, guidance_scale=8.5).images[0]

        image.save('generatedimage.png')
        img = ImageTk.PhotoImage(image)
        lmain.configure(image=img)

    trigger = ctk.CTkButton(art_frame, height=40, width=120, text_color="white", fg_color="blue", command=generate) 
    trigger.configure(text="Generate") 
    trigger.pack() 

    art_frame.pack()

def delete_pages():
    for frame in main_frame.winfo_children():
        frame.destroy()

def switch(page):
    delete_pages()
    page()

options_frame = tk.Frame(app, bg='#e8e4da', highlightbackground='black',highlightthickness=2)

home_btn = ctk.CTkButton(options_frame, text='Home', font=('Bold', 15), fg_color='#f9906f',hover_color='#f65b2a', command=lambda: switch(home_page))
home_btn.place(x=10, y=50)

frmimg_btn = ctk.CTkButton(options_frame, text='From Image', font=('Bold', 15), fg_color='#f9906f',hover_color='#f65b2a',  command=lambda: switch(still_image_page))
frmimg_btn.place(x=10, y=250)

realtime_btn = ctk.CTkButton(options_frame, text='Real Time', font=('Bold', 15), fg_color='#f9906f',hover_color='#f65b2a',  command=lambda: switch(real_time_page))
realtime_btn.place(x=10, y=500)

art_btn = ctk.CTkButton(options_frame, text='Create Art', font=('Bold', 15), fg_color='#f9906f',hover_color='#f65b2a',  command=lambda: switch(art_page))
art_btn.place(x=10, y=750)

options_frame.pack(side=tk.LEFT)
options_frame.pack_propagate(False)
options_frame.configure(width=200,height=1000)

main_frame = tk.Frame(app, bg='#c5bca3', highlightbackground='black',highlightthickness=2)
main_frame.pack(side=tk.LEFT)
main_frame.pack_propagate(False)
main_frame.configure(height=1000, width=1000)

mainimg = Image.open('miso_genie_logo.png')
tk_img = ImageTk.PhotoImage(mainimg)
image_label = tk.Label(main_frame, image=tk_img)
image_label.pack()

app.geometry("1200x1000")
app.title("Miso.Genie.AI")
app.resizable(False,False)
ctk.set_appearance_mode("dark")
#run
app.mainloop()