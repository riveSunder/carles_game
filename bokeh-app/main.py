from bokeh.io import curdoc

import numpy as np
import torch

from carle.carle.env import CARLE

import bokeh
import bokeh.io as bio
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

from bokeh.layouts import column, row
from bokeh.models import TextInput, Button, Paragraph
from bokeh.models import ColumnDataSource


env = CARLE(width=96, height=96)
obs = env.reset()
p = figure(plot_width=3*256, plot_height=3*256)

global my_period
my_period = 512

# add a circle renderer with x and y coordinates, size, color, and alpha

source = ColumnDataSource(data=dict(my_image=[obs.squeeze().cpu().numpy()]))
ColumnDataSource(data=dict(x=[1], y=[0]))
img = p.image(image='my_image',x=0, y=0, dw=256, dh=256, palette="Greys3", source=source)


button_go = Button(sizing_mode="stretch_width", label="Run >")     
button_slower = Button(sizing_mode="stretch_width",label="<< Slower")
button_faster = Button(sizing_mode="stretch_width",label="Faster >>")
button_reset = Button(sizing_mode="stretch_width",label="Reset")
message = Paragraph()

def update():
    
    action = 1.0 * (torch.rand(env.instances,1,env.action_height,env.action_width) < 0.05)
    
    obs, r, d, i = env.step(action)
     
    new_data = dict(my_image=[obs.squeeze().cpu().numpy()])
    
    source.stream(new_data, rollover=1)
    
    message.text = "Nominal update period = {} ms.".format(my_period)
    
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
    
def slower():
    
    global my_period
    my_period = min([my_period * 2, 8192])

def reset():
    obs = env.reset()
    
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
