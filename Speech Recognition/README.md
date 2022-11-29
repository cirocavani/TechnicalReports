# Speech Recognition

(TensorFlow tutorial)

**Simple audio recognition: Recognizing keywords**

<https://www.tensorflow.org/tutorials/audio/simple_audio>

<https://github.com/tensorflow/docs/blob/master/site/en/tutorials/audio/simple_audio.ipynb>

_This tutorial demonstrates how to pre-process audio files in the WAV format and build and train a basic automatic speech recognition (ASR) model for recognizing ten different words. You will use a portion of the Speech Commands dataset (Warden, 2018), which contains short (one-second or less) audio clips of commands, such as "down", "go", "left", "no", "right", "stop", "up" and "yes"._


## How to run

(Pluto Notebook)

Start [Pluto.jl](https://github.com/fonsp/Pluto.jl), open browser and run [`Simple audio recognition - Recognizing keywords.jl`](Simple%20audio%20recognition%20-%20Recognizing%20keywords.jl).

```sh
julia start_pluto.jl
# [ Info: Loading...
# [ Info: Listening on: 127.0.0.1:1234
# ┌ Info:
# └ Go to http://localhost:1234/ in your browser to start writing ~ have fun!
# ┌ Info:
# │ Press Ctrl+C in this terminal to stop Pluto
# └
```

## How to export HTML

(Pluto Slider Server)

Generate HTML using [PlutoSliderServer](https://github.com/JuliaPluto/PlutoSliderServer.jl) to run [`Simple audio recognition - Recognizing keywords.jl`](Simple%20audio%20recognition%20-%20Recognizing%20keywords.jl) and output [`Simple audio recognition - Recognizing keywords.html`](../site/2022-11-28/Speech%20Recognition.html).

```sh
julia export_html.jl
```
