# plotly utils.
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from os import path
from datetime import datetime

init_notebook_mode() # run at the start of every ipython notebook to use plotly.offline
                     # this injects the plotly.js source files into the notebook

def plot_xy(x, y, names=None, xlabel='x', ylabel='y', title=''):

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=xlabel,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=ylabel,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    if type(x) == list:
        xs = x
        ys = y
        if not names:
            names = ['line-' + str(i) for i in range(len(xs))]
    else:
        xs = [x]
        ys = [y]
        names = ['line']
    
    traces = []

    for (x, y, name) in zip(xs, ys, names):
        trace = go.Scatter(
            x = x,
            y = y,
            name = name
        )
        traces.append(trace)

    data = traces
    fig = go.Figure(data=data, layout=layout)
    disp = iplot(fig)


def plot_bar(xs, ys, names, xlabel='x', ylabel='y', title=''):

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=xlabel,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=ylabel,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    traces = []

    for (y, name) in zip(ys, names):
        trace = go.Bar(
            x = xs,
            y = y,
            name = name
        )
        traces.append(trace)

    data = traces
    fig = go.Figure(data=data, layout=layout)
    disp = iplot(fig, filename='mocha-gauge-caffe-plotbar-' + datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000'))
