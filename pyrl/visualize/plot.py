# plotly utils.
from pyrl.common import *
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from os import path
from datetime import datetime

init_notebook_mode() # run at the start of every ipython notebook to use plotly.offline
                     # this injects the plotly.js source files into the notebook

def plot_xy(x, y, names=None, error_ys=None, xlabel='x', ylabel='y', title=''):

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

    for (i, (x, y)) in enumerate(zip(xs, ys)):
        kwargs = {}
        if names:
            kwargs['name'] = names[i]
        if error_ys:
            kwargs['error_y'] = dict(
                type='data',     # or 'percent', 'sqrt', 'constant'
                array=error_ys[i],     # values of error bars
                visible=True
            )
        trace = go.Scatter(
            x = x,
            y = y,
            **kwargs
        )
        traces.append(trace)

    data = traces
    fig = go.Figure(data=data, layout=layout)
    disp = iplot(fig)


def plot_xye(x, y, err, names=None, xlabel='x', ylabel='y', title=''):

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
    traces_std = []

    std_colors = ['rgba(44,160,44,0.2)',
                    'rgba(214,39,40,0.2)',
                  ]

    for (i, (x, y, e)) in enumerate(zip(xs, ys, err)):
        kwargs = {}
        if names:
            kwargs['name'] = names[i]
        y = np.array(y)
        e = np.array(e)
        y_upper = list(y + e)
        y_lower = list(y - e)
        x = list(x)
        print 'y-upper', y_upper
        print 'y-lower', y_lower
        print 'y', y
        trace_std = go.Scatter(
            x = x + x[::-1],
            y = y_upper + y_lower[::-1],
            fill='tozerox',
            fillcolor=std_colors[i],
            line=go.Line(color='transparent'),
            showlegend=False,
            **kwargs
        )
        traces_std.append(trace_std)

        trace = go.Scatter(
            x = x,
            y = y,
            **kwargs
        )
        traces.append(trace)

    data = traces_std + traces
    fig = go.Figure(data=data, layout=layout)
    disp = iplot(fig)


def plot_bar(xs, ys, names, error_ys=None, xlabel='x', ylabel='y', title=''):

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

    for (i, y) in enumerate(ys):
        kwargs = {}
        if names:
            kwargs['name'] = names[i]
        if error_ys:
            kwargs['error_y'] = dict(
                type='data',     # or 'percent', 'sqrt', 'constant'
                array=error_ys[i],     # values of error bars
                visible=True
            )
        trace = go.Bar(
            x = xs[i],
            y = y,
            **kwargs
        )
        traces.append(trace)

    data = traces
    print 'data', data

    fig = go.Figure(data=data, layout=layout)
    # disp = iplot(fig, filename=datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000'))
    disp = iplot(fig) # offline mode, no need for filename.
