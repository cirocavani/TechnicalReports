cd(@__DIR__)

using Pkg

Pkg.activate(; temp=true)
Pkg.add("PlutoSliderServer")

using PlutoSliderServer

notebook_file = "Simple audio recognition - Recognizing keywords.jl"
PlutoSliderServer.export_notebook(notebook_file)
