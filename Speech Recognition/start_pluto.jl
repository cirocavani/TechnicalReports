cd(@__DIR__)

using Pkg

Pkg.activate(; temp=true)
Pkg.add("Pluto")

using Pluto

Pluto.run(;
    host="127.0.0.1",
    port=1234,
    launch_browser=false,
    require_secret_for_open_links=false,
    require_secret_for_access=false
)
