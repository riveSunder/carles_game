from bokeh.io import curdoc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from carle.carle.env import CARLE
from submission.agents import SubmissionAgent

import bokeh
import bokeh.io as bio
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

from bokeh.layouts import column, row
from bokeh.models import TextInput, Button, Paragraph
from bokeh.models import ColumnDataSource

env = CARLE()

agent = SubmissionAgent()

env.birth = [3]
env.survive = [0,2,3] 

global obs
obs = env.reset()

p = figure(plot_width=3*256, plot_height=3*256)

global my_period
my_period = 512

# add a circle renderer with x and y coordinates, size, color, and alpha

source = ColumnDataSource(data=dict(my_image=[obs.squeeze().cpu().numpy()]))
ColumnDataSource(data=dict(x=[1], y=[0]))
img = p.image(image='my_image',x=0, y=0, dw=256, dh=256, palette="Magma256", source=source)


button_go = Button(sizing_mode="stretch_width", label="Run >")     
button_slower = Button(sizing_mode="stretch_width",label="<< Slower")
button_faster = Button(sizing_mode="stretch_width",label="Faster >>")
button_reset = Button(sizing_mode="stretch_width",label="Reset")
message = Paragraph()

def update():
    global obs
    global stretch_pixel

    action = agent(obs)

    padded_action = stretch_pixel/2 + env.action_padding(action).squeeze()

    my_img = (padded_action*2 + obs.squeeze()).cpu().numpy()
    my_img[my_img > 3] = 3.0

    new_data = dict(my_image=[my_img])
    
    source.stream(new_data, rollover=1)
    
    message.text = "Nominal update period = {} ms.".format(my_period)
    obs, r, d, i = env.step(action)
    
def go():
   
    if button_go.label == "Run >":
        my_callback = curdoc().add_periodic_callback(update, my_period)
        button_go.label = "Pause"
        #curdoc().remove_periodic_callback(my_callback)
        
    else:
        curdoc().remove_periodic_callback(curdoc().session_callbacks[0])
        button_go.label = "Run >"

def faster():
    global my_period
    my_period = max([my_period * 0.5, 1])
    go()
    go()
    
def slower():
    global my_period
    my_period = min([my_period * 2, 8192])
    go()
    go()

def reset():
    global obs
    global stretch_pixel

    obs = env.reset()
    stretch_pixel = torch.zeros_like(obs).squeeze()
    stretch_pixel[0,0] = 3

    new_data = dict(my_image=[(stretch_pixel + obs.squeeze()).cpu().numpy()])

    source.stream(new_data, rollover=1)
    

reset()

button_go.on_click(go)
button_faster.on_click(faster)
button_slower.on_click(slower)
button_reset.on_click(reset)


control_layout = row(button_slower, button_go, button_faster, button_reset)
display_layout = row(p)
message_layout = row(message)

curdoc().add_root(display_layout)
curdoc().add_root(control_layout)
curdoc().add_root(message_layout)
