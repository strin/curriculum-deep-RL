import matplotlib.pyplot as plt
from cStringIO import StringIO
import IPython.display as display

class FrameSource(object):
    ''' an abstract interface for sources that generates
        frames
    '''
    def __init__(self):
        pass

    def render(self, frame_id):
        ''' render the current frame and save it to a np.ndarray buffer
        '''
        raise NotImplementedError()

    def generate(self, meta):
        ''' generate one next frame
            if source drains, the return None
        '''
        raise NotImplementedError()

    def terminated(self):
        ''' return True / False
        '''
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

class Frames(object):
    def __init__(self, source):
        '''
        - render_func: ()->np.ndarray is a method that Frames call to render the current frame in the buffer.
        '''
        self.source = source
        self.frame_id = -1
        self.keydown = None
        self.on_btn_next()

    def _encode_frame(self, data):
        fig = plt.figure()
        buffer = StringIO()
        plt.imshow(data, interpolation='none')
        fig.savefig(buffer, format='png')
        plt.close()
        buffer.reset()
        data_encoded = buffer.read().encode('base64')
        return "data:image/png;base64,{0}".format(data_encoded.replace('\n', ''))

    def render(self):
        data = self.source.render(self.frame_id)
        return self._encode_frame(data)

    def on_btn_next(self):
        if self.frame_id+1 >= len(self.source):
            if self.source.terminated():
                return
            else:
                self.source.generate({'keydown': self.keydown})
        self.frame_id += 1
        self.keydown = None

    def on_btn_prev(self):
        if self.frame_id > 0:
            self.frame_id -= 1
        self.keydown = None

    def on_key(self, keycode):
        self.keydown = keycode

    def html(self):
        disp = ['''
        <script>
            var kernel = IPython.notebook.kernel;
            kernel.execute('from ctypes import cast, py_object');
            function callback(msg) {
                var ret = msg.content.data['text/plain'];
                ret = ret.substring(1, ret.length-1);
                $('#frame_%(obj_id)d').attr('src', ret);
            };
        </script>
        <div>
            <img src='%(img)s' id='frame_%(obj_id)d'></img>
            <input id='btn_prev_%(obj_id)d' type='button' value='prev' onclick="javascript:
                kernel.execute('cast(%(obj_id)d, py_object).value.on_btn_prev()');
                kernel.execute('cast(%(obj_id)d, py_object).value.render()', {iopub: {output: callback}}, {silent: false});
                "></input>
            <input type='text' value='shortcut' id='shortcut_%(obj_id)d'></input>
            <input id='btn_next_%(obj_id)d' type='button' value='next' onclick="javascript:
                kernel.execute('cast(%(obj_id)d, py_object).value.on_btn_next()');
                kernel.execute('cast(%(obj_id)d, py_object).value.render()', {iopub: {output: callback}}, {silent: false});
                "></input>
        </div>
        <script>
        $('#shortcut_%(obj_id)d').keydown(function(event) {
            if(event.which == 37) { // left arrow
                $('#btn_prev_%(obj_id)d').click();
            }else if(event.which == 39) { // right arrow
                $('#btn_next_%(obj_id)d').click();
            }else{
                console.log('cast(%(obj_id)d, py_object).value.on_key(' + event.which.toString() + ')');
                kernel.execute('cast(%(obj_id)d, py_object).value.on_key(' + event.which.toString() + ')');
            }
            $('#shortcut_%(obj_id)d').val('' + event);
        });
        </script>
        ''' % dict(img=self.render(), obj_id=id(self))]
        return ''.join(disp)
