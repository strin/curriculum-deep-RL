# plotly utils.
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
