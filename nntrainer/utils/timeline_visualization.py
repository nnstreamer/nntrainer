# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 Hyungjun Seo <hyungjun.seo@samsung.com>
#
# @file timeline_visualization.py
# @date 14 April 2025
# @brief Visualize timeline using Plotly with log files
#
# @author Hyungjun Seo <hyungjun.seo@samsung.com>

import argparse

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
from collections import defaultdict


parser = argparse.ArgumentParser(description='arg for file_path')
parser.add_argument('log_file_path', type=str, help="Path to the file to read")  
args = parser.parse_args()  
log_file_path = args.log_file_path  
      
# prefill -> 0, token generation -> 1~
select_num = [i for i in range(0, 2)]

load_events = {}
inference_events = {}
rows = []
load_id_count = defaultdict(lambda: -1)
inference_id_count = defaultdict(lambda: -1)

event_types = ['LOAD_START', 'LOAD_END', 'INFERENCE_START', 'INFERENCE_END']

# log parsing
with open(log_file_path, 'r') as f:
  while True:
    line = f.readline()
    if not line: break
    
    parts = line.strip().split()
    if len(parts) != 3: continue
    event_type, id_, timestamp = parts
    if event_type not in event_types: continue
    timestamp = int(timestamp)
        
    if 'LOAD' in event_type:
      load_id_count[id_] += 1
      if (load_id_count[id_] // 2 not in select_num): continue
      id_ += "-" + str(load_id_count[id_] // 2)
      if 'START' in event_type:
        load_events[id_] = {'Start': timestamp}
      else:
        load_events[id_]['End'] = timestamp
      
    elif 'INFERENCE' in event_type:
      inference_id_count[id_] += 1
      if (inference_id_count[id_] // 2 not in select_num): continue
      id_ += "-" + str(inference_id_count[id_] // 2)
      if 'START' in event_type:
        inference_events[id_] = {'Start': timestamp}
      else:
        inference_events[id_]['End'] = timestamp

      
common_ids = set(load_events.keys()) & set(inference_events.keys())

pastels = [
  f"hsl(276, 100%, 70%)",f"hsl(248, 100%, 70%)",f"hsl(260, 100%, 70%)",f"hsl(50, 100%, 70%)",f"hsl(194, 100%, 70%)",
  f"hsl(125, 100%, 70%)",f"hsl(288, 100%, 70%)",f"hsl(72, 100%, 70%)",f"hsl(84, 100%, 70%)",f"hsl(124, 100%, 70%)",
  f"hsl(212, 100%, 70%)",f"hsl(333, 100%, 70%)",f"hsl(79, 100%, 70%)",f"hsl(29, 100%, 70%)",
]
def random_color():
  return f"hsl({random.randint(0, 360)}, {random.randint(50, 90)}%,{random.randint(30, 90)}%)"
  #return pastels[random.randint(0, 13)]

id_to_color = {id_: random_color() for id_ in common_ids}

# Task row
for id_, times in load_events.items():
  if 'Start' in times and 'End' in times:
    rows.append({
        'Task': 'LOAD',
        'ID': id_,
        'Start': pd.to_datetime(times['Start'], unit='ns'),
        'End': pd.to_datetime(times['End'], unit='ns'),
        'Description': f'LOAD ID: {id_}',
        'Color': id_to_color.get(id_, 'gray')
    })

for id_, times in inference_events.items():
  if 'Start' in times and 'End' in times:
    rows.append({
        'Task': 'INFERENCE',
        'ID': id_,
        'Start': pd.to_datetime(times['Start'], unit='ns'),
        'End': pd.to_datetime(times['End'], unit='ns'),
        'Description': f'INFERENCE ID: {id_}',
        'Color': id_to_color.get(id_, 'gray')
    })

# DataFrame
df = pd.DataFrame(rows)

# default timeline
# fig = px.timeline(
#     df, x_start="Start", x_end="End", y="Task", color="Color", hover_data=["Description"], custom_data=["ID"],
#     color_discrete_map=id_to_color
# )
fig = go.Figure()

for id_ in common_ids:
  load = load_events[id_]
  infer = inference_events[id_]
  
  start_load = pd.to_datetime(load['Start'], unit='us')
  end_load = pd.to_datetime(load['End'], unit='us')
  start_infer = pd.to_datetime(infer['Start'], unit='us')
  end_infer = pd.to_datetime(infer['End'], unit='us')
  y_load = [0.1, 0.2]
  y_infer = [0.0, 0.09]
  color = id_to_color[id_]
  
  fig.add_trace(go.Scatter(
      x=[start_load, start_load, end_load, end_load],
      y=[y_load[1], y_load[0], y_load[0], y_load[1]],
      fill='toself',
      mode='lines',
      line=dict(color=color, width=1),
      fillcolor=color,
      hoverinfo='text',
      text=f'LOAD ID: {id_},<br>Start: {start_load}<br>End: {end_load}<br>Duration: {(end_load - start_load) / 1000} ms',
      showlegend=False
  ))

  fig.add_trace(go.Scatter(
      x=[start_infer, start_infer, end_infer, end_infer],
      y=[y_infer[1], y_infer[0], y_infer[0], y_infer[1]],
      fill='toself',
      mode='lines',
      line=dict(color=color, width=1),
      fillcolor=color,
      hoverinfo='text',
      text=f'INFERENCE ID: {id_},<br>Start: {start_infer}<br>End: {end_infer}<br>Duration: {(start_infer - end_infer) / 1000} ms',
      showlegend=False
  ))


fig.update_yaxes(categoryorder="array", categoryarray=["INFERENCE", "LOAD"])
fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))
fig.update_layout(
    title="LOAD & INFERENCE Timeline",
    hoverlabel_align='left',
    xaxis_title="Time",
    yaxis_title="",
    dragmode="zoom",
    bargap = 0.02
)


# connection box
for id_ in common_ids:
  _, num = map(int, id_.split('-'))
  if (num not in select_num): continue
  
  load = load_events[id_]
  infer = inference_events[id_]

  start_load = pd.to_datetime(load['Start'], unit='us')
  end_load = pd.to_datetime(load['End'], unit='us')
  start_infer = pd.to_datetime(infer['Start'], unit='us')
  end_infer = pd.to_datetime(infer['End'], unit='us')

  color = id_to_color[id_]

  print(start_load, start_infer, end_infer, end_load)
  print([y_load[0], y_infer[1], y_infer[1], y_load[0]])
  fig.add_trace(go.Scatter(
      x=[start_load, start_infer, end_infer, end_load],
      y=[y_load[0], y_infer[1], y_infer[1], y_load[0]],
      fill='toself',
      mode='lines',
      line=dict(color=color, width=1),
      fillcolor=color,
      opacity=0.8,
      hoverinfo='skip',
      showlegend=False
  ))

pio.show(fig, config={'scrollZoom': True})
