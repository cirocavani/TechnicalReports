### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a08f9487-6a91-4a80-a479-4fd3a7e4d0c7
begin
	using BSON
	using CairoMakie
	using Downloads
	using DSP
	using Flux
	using MLUtils
	using OneHotArrays
	using Random
	using Statistics
	using WAV
	using ZipFile

	using PlutoUI

	CairoMakie.activate!()

	Random.seed!(42)
end;

# ╔═╡ 3037a996-6acc-11ed-2c2f-8f490121dc68
md"""
# Simple audio recognition: Recognizing keywords

<https://www.tensorflow.org/tutorials/audio/simple_audio>

_This tutorial demonstrates how to preprocess audio files in the WAV format and build and train a basic automatic speech recognition (ASR) model for recognizing ten different words. You will use a portion of the Speech Commands dataset (Warden, 2018), which contains short (one-second or less) audio clips of commands, such as "down", "go", "left", "no", "right", "stop", "up" and "yes"._
"""

# ╔═╡ b2eef973-24f1-47c5-9ed6-5b0ad5ee3442
TableOfContents()

# ╔═╡ 6c87265a-feb9-4e3c-9076-535f81b3fc5a
md"""## Import the mini Speech Commands dataset"""

# ╔═╡ 2d5ef07d-9890-431b-a8fc-0f62bd6c5f29
begin
	function extract_dataset(zipfile_path::String, output_dir::String)::Int
		n_files = 0
		zipfile_reader = ZipFile.Reader(zipfile_path)
		try
			for file ∈ zipfile_reader.files
				!startswith(file.name, "__MACOSX/") || continue
				output_path = joinpath(output_dir, file.name)
				if endswith(output_path, "/")
					mkdir(output_path)
				else
					write(output_path, read(file))
					n_files += 1
				end
			end
		finally
			close(zipfile_reader)
		end

		return n_files
	end

	function get_dataset(
		data_dir::String;
		overwrite = false,
	)::String
		isdir(data_dir) || mkpath(data_dir)

		dataset_dir = joinpath(data_dir, "mini_speech_commands")
		if isdir(dataset_dir)
			overwrite || return dataset_dir
			rm(dataset_dir; recursive = true)
		end
		
		dataset_file = "mini_speech_commands.zip"
		dataset_file_path = joinpath(data_dir, dataset_file)
		found = isfile(dataset_file_path)
		if found && overwrite
			rm(dataset_file_path)
		end
		if !found || overwrite
			dataset_url = "http://storage.googleapis.com/download.tensorflow.org/data/$(dataset_file)"
			Downloads.download(dataset_url, dataset_file_path)
		end

		n_files = extract_dataset(dataset_file_path, data_dir)
		n_files == 8001 || error("Invalid number of files: $(n_files)")

		isdir(dataset_dir) || error("Dataset folder not found: $(dataset_dir)")

		return dataset_dir
	end

	data_dir = "data"
	dataset_dir = get_dataset(data_dir; overwrite=false)
end

# ╔═╡ 687d6db7-4fc8-4425-8b33-2eaabd985a4d
commands = filter(!=("README.md"), readdir(dataset_dir; sort = true))

# ╔═╡ 1e065d4d-beb8-49f8-972a-b39bfc2bf340
onehot_labels = onehotbatch(["go", "go", "no", "down"], commands)

# ╔═╡ c38e7c4e-e6aa-4f4b-9954-2cb1bcd2059d
begin
	function read_audio_data(audio_file::String)::Vector{Float32}
		x, f = wavread(audio_file)
		f == 16000f0 || error("[$(audio_file)] Invalid frequency: $(f)")
		size(x, 2) == 1 || error("[$(audio_file)] Invalid channels: $(size(x, 2))")
		x = dropdims(x; dims = 2) # (samples, channels=1) -> (samples)
		n = length(x)
		if n != 16000 # 1 second == 16000 samples
			resize!(x, 16000)
			n < 16000 && (x[n+1:end] .= 0.)
		end
		x = Float32.(x)
	
		return x
	end

	function load_dataset(
		dataset_dir::String,
	)::Tuple{
		Vector{Vector{Float32}},
		OneHotMatrix,
		Vector{String},
		Vector{Int},
		Vector{String},
	}
		label_names = filter(!=("README.md"), readdir(dataset_dir; sort = true))
		label_indices = Vector{Int}()
		data_files = Vector{String}()
		X = Vector{Vector{Float32}}()
		Y = Vector{String}()
		for label_name ∈ label_names
			push!(label_indices, length(Y) + 1)

			audio_files = readdir(joinpath(dataset_dir, label_name); join = true)
			audio_data = map(read_audio_data, audio_files)
			audio_label = fill(label_name, length(audio_data))
			append!(data_files, audio_files)
			append!(X, audio_data)
			append!(Y, audio_label)
		end

		Y = onehotbatch(Y, label_names)::OneHotMatrix

		return X, Y, label_names, label_indices, data_files
	end

	dataset_X, dataset_Y, dataset_labels, dataset_indices, dataset_files =
		load_dataset(dataset_dir)
	
	@assert size(dataset_X) == (8000,)
	@assert all(length.(dataset_X) .== 16000)
	@assert size(dataset_Y) == (8, 8000)
	@assert size(dataset_labels) == (8,)
	@assert size(dataset_indices) == (8,)
	@assert size(dataset_files) == (8000,)
end;

# ╔═╡ c438523e-8aab-4e47-8cb6-abe589ae887d
begin
	function plot_waveform(
		waveform::Vector{Float32};
		plot_parent::Union{GridPosition,Nothing} = nothing,
		title::Union{AbstractString,Nothing} = nothing,
	)::Figure
		if isnothing(plot_parent)
			f = Figure(; resolution = (800, 350))
			plot_parent = f[1, 1]
		end
		ax = Axis(
			plot_parent;
			limits = (0, 16000, -1.0, 1.0),
			xticks = 0:1600:16000,
			xtickformat = x -> string.(round.(x / 16000; digits = 2)),
			yticks = -1.0:0.2:1.0,
		)
		isnothing(title) || (ax.title = title)
		lines!(ax, waveform)

		return current_figure()
	end

	plot_waveform(dataset_X[1]; title = "$(dataset_labels[1]) 1")
end

# ╔═╡ 72c3fa6b-8595-4d99-b923-9dc8d614b0d3
let
	f = Figure(; resolution = (1200, 2000))
	for (label_index, label_name) = enumerate(dataset_labels)
		label_grid = f[label_index, 1] = GridLayout()
		Label(label_grid[1, 1:3], label_name; halign = :left)
		for j = 1:3
			plot_waveform(
				dataset_X[dataset_indices[label_index] + j - 1];
				plot_parent=label_grid[2, j],
			)
		end
	end
	f
end

# ╔═╡ b08b7f4b-925d-4e7b-8d40-ef15dc8da448
@bind play_command PlutoUI.Button("Play Random")

# ╔═╡ 3457dc29-a4f1-4595-bea5-120f76ac2b17
let
	play_command

	data_index = rand(1:length(dataset_X))
	waveform = dataset_X[data_index]
	label = onecold(dataset_Y[:, data_index], dataset_labels)
	title = "$label (index=$data_index)"

	plot_waveform(waveform; title)

	wavplay(waveform, 16000)

	current_figure()
end

# ╔═╡ 99c054ad-5bc0-4aec-93d4-d26c20cd1b10
md"""## Convert waveforms to spectrograms"""

# ╔═╡ 5f5858eb-bf40-4b54-9512-5719e2a3cfd5
begin
	function get_spectrogram(waveform::Vector{Float32})::Matrix{Float32}
		return abs.(stft(waveform, 255, 128))
	end

	waveform_example = dataset_X[1]
	spectrogram_example = get_spectrogram(waveform_example)

	@assert size(spectrogram_example) == (129, 124)

	spectrogram_example
end

# ╔═╡ 1a441922-9f28-4628-9070-ac3a5d755790
plot_waveform(waveform_example)

# ╔═╡ 051ab7d8-2075-4f40-a4b8-e27fba63b80e
begin
	function plot_spectrogram(
		spectrogram::Matrix{Float32};
		plot_parent::Union{GridPosition,Nothing} = nothing,
		title::Union{AbstractString,Nothing} = nothing,
	)::Figure
		log_spec = log.(spectrogram' .+ eps(Float32))
		height, width = size(log_spec)
		X = collect(range(0, height * width; length = width))
		Y = collect(0.0:height-1)

		if isnothing(plot_parent)
			f = Figure(; resolution = (800, 350))
			plot_parent = f[1, 1]
		end
		ax = Axis(
			plot_parent;
			limits = ((0, 16000), (0, 123)),
			xticks = 0:1600:16000,
			xtickformat = x -> string.(round.(x / 16000; digits = 2)),
			yticks = 0:20:123,
		)
		isnothing(title) || (ax.title = title)
		heatmap!(ax, X, Y, log_spec)

		return current_figure()
	end
	
	plot_spectrogram(spectrogram_example)
end

# ╔═╡ 1e974e18-76ad-49ee-bf23-f3078402021b
let
	f = Figure(; resolution = (1200, 2000))
	for (label_index, label_name) = enumerate(dataset_labels)
		label_grid = f[label_index, 1] = GridLayout()
		Label(label_grid[1, 1:3], label_name; halign = :left)
		for j = 1:3
			waveform = dataset_X[dataset_indices[label_index] + j - 1]
			spectrogram = get_spectrogram(waveform)
			plot_spectrogram(
				spectrogram;
				plot_parent = label_grid[2, j],
			)
		end
	end
	f
end

# ╔═╡ 8bed4018-de19-4266-af39-d70f8215fbc9
@bind play_command2 PlutoUI.Button("Play Random")

# ╔═╡ 19a42016-d911-4fca-b5eb-abb802740728
let
	play_command2

	data_index = rand(1:length(dataset_X))
	waveform = dataset_X[data_index]
	spectrogram = get_spectrogram(waveform)
	label = onecold(dataset_Y[:, data_index], dataset_labels)
	title = "$label (index=$data_index)"
	
	f = Figure(; resolution = (800, 700))
	Label(f[1,1:3], title; halign = :center)
	plot_waveform(waveform; plot_parent=f[2,1:3], title = "Waveform")
	plot_spectrogram(spectrogram; plot_parent=f[3,1:3], title = "Spectrogram")

	wavplay(waveform, 16000)

	f
end

# ╔═╡ f5c3d15e-0110-4c1e-a9b0-c45a6192efc2
md"""## Build and train the model"""

# ╔═╡ b756ab36-1178-4748-b10c-62fc6b6293db
dataset_X_spec = mapobs(x -> unsqueeze(get_spectrogram(x); dims = 3), dataset_X);

# ╔═╡ 70fa026a-7c07-417e-902d-7d890762973f
dataset_train, dataset_valid, dataset_test = 
	splitobs(
		shuffleobs((dataset_X_spec, dataset_Y));
		at = (0.8, 0.1),
	);

# ╔═╡ 25733f1b-7ec7-4786-91a6-77dc4c1efd24
train_data = DataLoader(
	dataset_train;
	batchsize = 64,
	shuffle = true,
	collate = true,
);

# ╔═╡ e4b07f3e-05f0-4987-bbfa-abcc2762d9b3
x1, y1 = first(train_data);

# ╔═╡ 51db35f2-bf8b-47f1-8727-a35f90b58db4
@assert size(x1) == (129, 124, 1, 64)

# ╔═╡ 09f55e73-a120-4b7d-a66c-161728a77cc3
x1[:, :, :, 1]

# ╔═╡ 2efb7844-f23c-43e0-8b42-3e84788b669c
@assert size(y1) == (8, 64)

# ╔═╡ c52286ac-abd4-4e99-bb9c-196775258816
y1

# ╔═╡ 99ddfadd-e56d-43fd-bfd6-2ef05df939d7
md"""
Downsample the input.

<https://www.tensorflow.org/api_docs/python/tf/keras/layers/Resizing>
"""

# ╔═╡ 4b083413-93e4-4439-8c5f-ae38606c55b2
let
	layer = Upsample(:bilinear; size = (32, 32))

	input = x1
	output = layer(input)

	@assert size(input) == (129, 124, 1, 64)
	@assert size(output) == (32, 32, 1, 64)

	output[:, :, :, 1]
end

# ╔═╡ 39dd2002-847e-4a74-9816-20e43ab69cb6
md"""
Normalize.

<https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization>
"""

# ╔═╡ 3347548c-a467-40e5-81d3-cc9ee70b0b4b
# Just for testing (closure is better for model saving)
function Normalization(μ::T, σ::T) where {T<:Real}
	!iszero(σ) || (σ = eps(T))
	return x -> @.((x - μ) / σ)
end

# ╔═╡ 6989a48d-03d2-476d-83c7-073c0bb98ab2
let
	input = x1

	# σ == 0 -> no infs/nans
	layer = Normalization(0.f0, 0.f0)
	output = layer(input)
	@assert eltype(output) === Float32
	@assert findfirst(∈([Inf32,-Inf32,NaN32]), output) === nothing

	# identity μ=0 σ=1
	layer = Normalization(0.f0, 1.f0)
	output = layer(input)
	@assert input ≈ output
end

# ╔═╡ ce42db49-3c2c-4e64-a52b-ad18f9844afb
begin
	function normalization_params(data)
		# data -> Vector{<:AbstractArray{Float32}}
		# for all i, length(data[i]) = K where K is a constant.
		# @assert length(Set(length.(data))) == 1
		μ = mean(mean.(data))
		σ = sqrt(mean(var.(data; mean = μ, corrected = false)))
		return μ, σ
	end

	sample_data = dataset_X_spec[rand(1:length(dataset_X_spec), 10)]

	A = hcat(sample_data...)
	μ_A = mean(A)
	σ_A = std(A; mean = μ_A, corrected = false)

	μ₀, σ₀ = normalization_params(sample_data)

	@assert μ₀ ≈ μ_A
	@assert σ₀ ≈ σ_A

	(; μ₀, σ₀)
end

# ╔═╡ 2261341f-9c8b-4f61-abff-6aa5a81fc337
μ_train, σ_train = normalization_params(dataset_train[1])

# ╔═╡ 4acd8856-c99f-415e-afe9-7b950cb4d4fc
let
	layer = Normalization(μ_train, σ_train)

	input = x1
	output = layer(input)

	@assert size(input) == (129, 124, 1, 64)
	@assert size(output) == (129, 124, 1, 64)

	μᵢ = mean(input)
	σᵢ = std(input; mean = μᵢ, corrected = false)

	μₒ = mean(output)
	σₒ = std(output; mean = μₒ, corrected = false)

	# (; μ_train, σ_train, μ = μᵢ => μₒ, σ = σᵢ => σₒ)
	md"""
	| Measure | Value       |
	|:--------|:------------|
	| μ_train | $μ_train    |
	| σ_train | $σ_train    |
	| μ       | $(μᵢ => μₒ) |
	| σ       | $(σᵢ => σₒ) |
	"""
end

# ╔═╡ cccc14ce-0e5d-4039-857a-69820cec2d49
model = Chain(
	# input WHCN 129x124x1[x64]
	Upsample(:bilinear; size = (32, 32)), # 129x124x1 -> 32x32x1
	x -> @.((x - μ_train) / σ_train),     # (closure better for model saving)
	Conv((3, 3), 1 => 32, relu),          # 32x32x1 -> 30x30x32
	Conv((3, 3), 32 => 64, relu),         # 30x30x32 -> 28x28x64
	MaxPool((2, 2)),                      # 28x28x64 -> 14x14x64
	Dropout(0.25),
	Flux.flatten,                         # 14x14x64 -> 12544
	Dense(12544 => 128, relu),            # 12544 -> 128
	Dropout(0.5),
	Dense(128 => length(dataset_labels)), # 128 -> 8
)

# ╔═╡ 32507e28-8796-4d78-ae87-cd73fe156e08
ŷ1 = model(x1)

# ╔═╡ 4318adee-2c07-4dfc-901b-c8c229c1869c
loss1 = Flux.logitcrossentropy(ŷ1, y1)

# ╔═╡ bff95353-bbdc-4617-8be5-6f6d5f6e82ab
begin
	function loss(model, x, y)
		ŷ = model(x)
		L = Flux.logitcrossentropy(ŷ, y)
		return L
	end

	function loss(model, data)
		x, y = data
		L = loss(model, batch(x), y)
		return L
	end
	
	loss(model, dataset_valid)
end

# ╔═╡ 70b0f14a-7c28-461d-8e6f-53051f3030ce
begin
	function accuracy(model, x, y)
		ŷ_label_index = onecold(model(x))
		y_label_index = onecold(y)
		acc = mean(ŷ_label_index .== y_label_index)
		return acc
	end

	function accuracy(model, data)
		x, y = data
		acc = accuracy(model, batch(x), y)
		return acc
	end
	
	accuracy(model, dataset_valid)
end

# ╔═╡ 5c22cbda-e999-4857-86c7-427fcacdac06
history = let
	params = Flux.params(model)
	opt = Flux.Adam()

	history = Dict{String,Vector{Float64}}()
	history["loss"] = Vector{Float64}()
	history["accuracy"] = Vector{Float64}()
	history["val_loss"] = Vector{Float64}()
	history["val_accuracy"] = Vector{Float64}()
	
	n_epochs = 10
	for epoch = 1:n_epochs
		@info "Epoch $epoch/$n_epochs"
		Flux.train!(params, train_data, opt) do x, y
			loss(model, x, y)
		end
		
		train_loss = mean(loss(model, x, y) for (x, y) ∈ train_data)
		train_accuracy = mean(accuracy(model, x, y) for (x, y) ∈ train_data)
		valid_loss = loss(model, dataset_valid)
		valid_accuracy = accuracy(model, dataset_valid)
		@info "loss: $train_loss - accuracy: $train_accuracy - val_loss: $valid_loss - val_accuracy: $valid_accuracy"
		push!(history["loss"], train_loss)
		push!(history["accuracy"], train_accuracy)
		push!(history["val_loss"], valid_loss)
		push!(history["val_accuracy"], valid_accuracy)
	end

	history
end;

# ╔═╡ d7af0058-924f-41af-ad21-ba910059b111
let
	train_loss = history["loss"]
	valid_loss = history["val_loss"]

	train_accuracy = history["accuracy"]
	valid_accuracy = history["val_accuracy"]
	
	f = Figure(; resolution = (800, 350))
	ax1 = Axis(
		f[1, 1];
		xlabel = "Epoch",
		ylabel = "Loss [Cross Entropy]",
		limits = (1, 10, 0, 1.75),
		xticks = 1:10,
		yticks = 0:0.25:1.75,
	)
	lines!(ax1, train_loss; label = "loss")
	lines!(ax1, valid_loss; label = "val_loss")
	axislegend(ax1)
	
	ax2 = Axis(
		f[1, 2];
		xlabel = "Epoch",
		ylabel = "Accuracy [%]",
		limits = (1, 10, 0, 100),
		xticks = 1:10,
		yticks = 0:20:100,
	)
	lines!(ax2, 100 .* train_accuracy; label = "accuracy")
	lines!(ax2, 100 .* valid_accuracy; label = "val_accuracy")
	axislegend(ax2; position = :rb)
	
	f
end

# ╔═╡ 75a60b92-8278-4d10-9bfb-dd14fabc4fc8
md"""## Evaluate the model performance"""

# ╔═╡ b4c3040e-13b0-48ee-8652-57f058440529
let
	f = x -> round(Float64(x); digits = 4)
	train_loss = f(mean(loss(model, x, y) for (x, y) ∈ train_data))
	valid_loss = f(loss(model, dataset_valid))
	test_loss = f(loss(model, dataset_test))

	g = x -> 100 * x
	train_acc = g(mean(accuracy(model, x, y) for (x, y) ∈ train_data))
	valid_acc = g(accuracy(model, dataset_valid))
	test_acc = g(accuracy(model, dataset_test))

	md"""
	| Dataset    | Loss [Cross Entropy] | Accuracy [%] |
	|:-----------|---------------------:|-------------:|
	| Train      |          $train_loss |   $train_acc |
	| Validation |          $valid_loss |   $valid_acc |
	| Test       |           $test_loss |    $test_acc |
	"""
end

# ╔═╡ 7fd7b3d7-d3bd-46e0-89b9-81b6a81c4907
md"""### Display a confusion matrix"""

# ╔═╡ d012ed96-5112-413d-b918-df25fac3ccae
begin
	function confusion_matrix(model, x, y)
		ŷ = model(x)
		y_label_index = onecold(y)
		ŷ_label_index = onecold(ŷ)
		m = zeros(Int, (size(y, 1), size(ŷ, 1)))
		for (y_label, ŷ_label) ∈ zip(y_label_index, ŷ_label_index)
			m[y_label, ŷ_label] += 1
		end
		return m
	end

	function confusion_matrix(model, data)
		x, y = data
		m = confusion_matrix(model, batch(x), y)
		return m
	end

	confusion_mtx = confusion_matrix(model, dataset_test)
end

# ╔═╡ a3b61da5-7738-416f-98cb-9a76f83af82a
let
	M = reverse(confusion_mtx; dims = 1)
	n, m = size(M)
	v = vec(M)
	μ = mean(v)
	
	f = Figure()
	ax = Axis(
		f[1, 1];
		title = "Confusion Matrix",
		xlabel = "Prediction",
		ylabel = "Label",
		xticks = 1:m,
		xtickformat = x -> dataset_labels[Int.(x)],
		yticks = 1:n,
		ytickformat = y -> dataset_labels[-Int.(y) .+ (n + 1)],
		xaxisposition = :top,
	)
	heatmap!(ax, M'; colormap = :seaborn_rocket_gradient)
	text!(
		ax,
        string.(v);
        position = [Point2f(x, y) for x = 1:m for y = 1:n],
        align = (:center, :center),
        color = ifelse.(v .< μ, :white, :black),
        textsize = 12,
	)
	
	f
end

# ╔═╡ 9c7a5f70-5897-4867-839d-dbad86773b85
md"""## Run inference on an audio file"""

# ╔═╡ ef21c525-ba3f-43a5-a0f6-79f83618a952
begin
	function run_model(model, waveform::Vector{Float32})::Vector{Float32}
		spectrogram = get_spectrogram(waveform)
		input = reshape(spectrogram, size(spectrogram)..., 1, 1)
		output = model(input)
		probs = softmax(vec(output))
		return probs
	end

	function run_model(model, wav_file::String)::Vector{Float32}
		waveform = read_audio_data(wav_file)
		probs = run_model(model, waveform)
		return probs
	end

	function print_label_probs(label_names, probs)
		label_probs = collect(zip(label_names, probs))
		best_labels = sort(label_probs; by=last, rev=true)
		for (i, (label_name, label_prob)) = enumerate(best_labels)
			label_prob = round(100 * label_prob; digits = 2)
			println("$i: $(rpad(label_name, 5)) $(lpad(label_prob, 5))%")
		end
	end

	probs_example = run_model(
		model,
		joinpath(dataset_dir, "no/01bb6a2a_nohash_0.wav"),
	)

	print_label_probs(dataset_labels, probs_example)

	probs_example
end

# ╔═╡ 9ea2cc9c-a6e3-45f7-b2b2-7823af10cf85
begin
	function plot_classification(
		labels::Vector{String},
		probs::Vector{Float32};
		plot_parent::Union{GridPosition,Nothing} = nothing,
		title::Union{AbstractString,Nothing} = nothing,
	)::Figure
		if isnothing(plot_parent)
			f = Figure(; resolution = (800, 350))
			plot_parent = f[1, 1]
		end
		ax = Axis(
			plot_parent;
			xlabel = "Labels",
			ylabel = "Probability [%]",
			xticks = 1:length(labels),
			xtickformat = x -> labels[Int.(x)],
			yticks = 0:20:100,
			limits = (0, length(labels)+1, 0, 100),
		)
		isnothing(title) || (ax.title = title)
		barplot!(ax, 100 .* probs)

		return current_figure()
	end

	plot_classification(
		dataset_labels,
		probs_example;
		title="no/01bb6a2a_nohash_0.wav",
	)
end

# ╔═╡ 16ae8d32-d54b-441b-b544-745a2c5ea314
let
	f = Figure(; resolution = (1200, 2000))
	for (label_index, label_name) = enumerate(dataset_labels)
		label_grid = f[label_index, 1] = GridLayout()
		Label(label_grid[1, 1:3], label_name; halign = :left)
		for j = 1:3
			wav_file = dataset_files[dataset_indices[label_index] + j - 1]
			probs = run_model(model, wav_file)
			plot_classification(
				dataset_labels,
				probs;
				plot_parent = label_grid[2, j],
			)
		end
	end
	f
end

# ╔═╡ c011a460-2276-4704-ae65-fd7098f6e21b
@bind inference PlutoUI.Button("Inference")

# ╔═╡ ce4b7434-f0b1-42b2-9ac6-7bf437a3f9f0
let
	inference

	wav_file = rand(dataset_files)
	waveform = read_audio_data(wav_file)
	probs = run_model(model, waveform)

	label = basename(dirname(wav_file))

	f = Figure(; resolution = (800, 700))
	Label(f[1,1:3], label; halign = :center)
	plot_waveform(waveform; plot_parent=f[2, 1:3])
	plot_classification(dataset_labels, probs; plot_parent=f[3, 1:3])

	wavplay(waveform, 16000)
	
	f
end

# ╔═╡ ca047acc-c51f-4975-977f-eda50b12822c
md"""## Export the model with preprocessing"""

# ╔═╡ 160deb3d-7673-46cf-bda2-cb51f51ba12d
model_file = joinpath(data_dir, "saved_model.bson")

# ╔═╡ 78049b33-4270-46e5-a5d1-9601582be47c
!isfile(model_file) || rm(model_file)

# ╔═╡ b9de6ffc-b99a-403b-9c38-f2d53b611048
BSON.@save(model_file, model)

# ╔═╡ 656f91e1-8335-45a5-a8b2-12725888c7e1
# test load using Pluto workspace module (default module would be Main) 
BSON.load(model_file, @__MODULE__)

# ╔═╡ 0f8aad0b-a1ba-4e59-9d71-5b3be612c500
inference_app_code = raw"""
using BSON
using DSP
using Flux
using Random
using WAV

function read_model(model_file::String)::Chain
	return BSON.load(model_file, @__MODULE__)[:model]
end

function read_audio_data(audio_file::String)::Vector{Float32}
	x, f = wavread(audio_file)
	f == 16000f0 || error("[$(audio_file)] Invalid frequency: $(f)")
	size(x, 2) == 1 || error("[$(audio_file)] Invalid channels: $(size(x, 2))")
	x = dropdims(x; dims = 2) # (samples, channels=1) -> (samples)
	n = length(x)
	if n != 16000 # 1 second == 16000 samples
		resize!(x, 16000)
		n < 16000 && (x[n+1:end] .= 0.)
	end
	x = Float32.(x)

	return x
end

function get_spectrogram(waveform::Vector{Float32})::Matrix{Float32}
	return abs.(stft(waveform, 255, 128))
end

function run_model(model, waveform::Vector{Float32})::Vector{Float32}
	spectrogram = get_spectrogram(waveform)
	input = reshape(spectrogram, size(spectrogram)..., 1, 1)
	output = model(input)
	probs = softmax(vec(output))
	return probs
end

function run_model(model, wav_file::String)::Vector{Float32}
	waveform = read_audio_data(wav_file)
	probs = run_model(model, waveform)
	return probs
end

function print_label_probs(label_names, probs)
	label_probs = collect(zip(label_names, probs))
	best_labels = sort(label_probs; by=last, rev=true)
	for (i, (label_name, label_prob)) = enumerate(best_labels)
		label_prob = round(100 * label_prob; digits = 2)
		println("$i: $(rpad(label_name, 5)) $(lpad(label_prob, 5))%")
	end
end

if length(ARGS) < 2
	println("Usage: julia run_model.jl <model_file> <wav_file>")
	exit(1)
end

model_file = ARGS[1]
wav_file = ARGS[2]

if !isfile(model_file)
	println("Model file not found: $model_file")
	exit(1)
end
if !isfile(wav_file)
	println("WAV file not found: $wav_file")
	exit(1)
end

println("Model file = $model_file")
println("WAV file = $wav_file")
println()

waveform = read_audio_data(wav_file)
wavplay(waveform, 16000)

model = read_model(model_file)
probs = run_model(model, waveform)

label_names = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
print_label_probs(label_names, probs)
""";

# ╔═╡ e38d4066-b520-48bb-b238-900d658eccab
inference_app_file = joinpath(data_dir, "run_model.jl")

# ╔═╡ e47e9820-09ca-4b46-9f98-f8d38f0e3668
!isfile(inference_app_file) || rm(inference_app_file)

# ╔═╡ 4956e408-4b48-438f-b785-1cd67b66708b
write(inference_app_file, inference_app_code)

# ╔═╡ a8d73067-e78c-4a4d-a7a1-1324275ab192
@bind inference2 PlutoUI.Button("Inference")

# ╔═╡ 7f14c560-9a9e-4588-9dbc-352929569d95
let
	inference2

	wav_file = rand(dataset_files)
	run(`julia --project=$(Base.active_project()) $inference_app_file $model_file $wav_file`)
end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
WAV = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"
ZipFile = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"

[compat]
BSON = "~0.3.6"
CairoMakie = "~0.9.3"
DSP = "~0.7.7"
Flux = "~0.13.8"
MLUtils = "~0.3.1"
OneHotArrays = "~0.2.0"
PlutoUI = "~0.7.48"
WAV = "~1.2.0"
ZipFile = "~0.10.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[AbstractTrees]]
git-tree-sha1 = "52b3b436f8f73133d7bc3a6c71ee7ed6ab2ab754"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.3"

[[Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "eb7a1342ff77f4f9b6552605f27fd432745a53a3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.22"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[BSON]]
git-tree-sha1 = "86e9781ac28f4e80e9b98f7f96eae21891332ac2"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.6"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "7fe6d92c4f281cf4ca6f2fba0ce7b299742da7ca"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.37"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "49549e2c28ffb9cc77b3689dc10e46e6271e9452"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.12.0"

[[Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "SnoopPrecompile"]
git-tree-sha1 = "20bd6ace08bb83bf5579e8dfb0b1e23e33518b04"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.9.3"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "0c8c8887763f42583e1206ee35413a43c91e2623"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.45.0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "aaabba4ce1b7f8a9b34c015053d3b1edf60fa49c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.4.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "4ba2a190a9d05a36e8c26182eb1ba06cd12c1051"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.7"

[[DataAPI]]
git-tree-sha1 = "e08915633fcb3ea83bf9d6126292e5bc5c739922"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.13.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "c5b6685d53f933c11404a3ae9822afe30d522494"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.12.2"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "a7756d098cbabec6b3ac44f369f74915e8cfd70a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.79"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "802bfc139833d2ba893dd9e62ba1767c88d708ae"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.5"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Flux]]
deps = ["Adapt", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "OneHotArrays", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "fdbac308c552f749122b0ad8d7cdca02ac131531"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.8"

[[FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "10fa12fe96e4d76acfa738f4df2126589a67374f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.33"

[[FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "38a92e40157100e796690421e34a11c107205c86"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "a2657dd0f3e8a61dbe70fc7c122038bd33790af5"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.3.0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "45d7deaf05cbb44116ba785d147c518ab46352d7"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.5.0"

[[GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "76f70a337a153c1632104af19d29023dbb6f30dd"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.6"

[[GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "fe9aea4ed3ec6afdfbeb5a4f39a2208909b162a6"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.5"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fb83fbe02fe57f2c068013aa94bcdf6760d3a7a7"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+1"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "678d136003ed5bceaab05cf64519e3f956ffa4ba"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.9.1"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "2e99184fca5eb6f075944b04c22edec29beb4778"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.7"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "842dd89a6cb75e02e85fdd75c760cdc43f5d6863"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.6"

[[IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "3f91cd3f56ea48d4d2a75c2a65455c5fc74fa347"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.3"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[MLStyle]]
git-tree-sha1 = "060ef7956fef2dc06b0e63b294f7dbfbcbdc7ea2"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.16"

[[MLUtils]]
deps = ["ChainRulesCore", "DataAPI", "DelimitedFiles", "FLoops", "FoldsThreads", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "82c1104919d664ab1024663ad851701415300c5f"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.3.1"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "MiniQhull", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "d3b9553c2f5e0ca588e4395a9508cef024bd9e8a"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.18.3"

[[MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "c1885d865632e7f37e5a1489a164f44c54fb80c9"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.5.2"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test", "UnicodeFun"]
git-tree-sha1 = "f04120d9adf4f49be242db0b905bea0be32198d1"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.4"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "4d5917a26ca33c66c8e5ca3247bd163624d35493"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.3"

[[MiniQhull]]
deps = ["QhullMiniWrapper_jll"]
git-tree-sha1 = "9dc837d180ee49eeb7c8b77bb1c860452634b0d1"
uuid = "978d7f02-9e05-4691-894f-ae31a51d76ca"
version = "0.4.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "00bcfcea7b2063807fdcab2e0ce86ef00b8b8000"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.10"

[[NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "4429261364c5ea5b7308aecaa10e803ace101631"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.4"

[[NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "GPUArraysCore", "LinearAlgebra", "MLUtils", "NNlib"]
git-tree-sha1 = "aee0130122fa7c1f3d394231376f07869f1e097c"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.0"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6e9dba33f9f2c44e08a020b0caf6903be540004"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.19+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "8a9102cb805df46fc3d6effdc2917f09b0215c0b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.10"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "84a314e3926ba9ec66ac097e3635e270986b0f10"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.9+0"

[[Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "b64719e8b4504983c7fca6cc9db3ebc8acc2a4d6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.1"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3010a6dd6ad4c7384d2f38c58fa8172797d879c1"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[QhullMiniWrapper_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Qhull_jll"]
git-tree-sha1 = "607cf73c03f8a9f83b36db0b86a3a9c14179621f"
uuid = "460c41e3-6112-5d7f-b78c-b6823adb3f2d"
version = "1.0.0+1"

[[Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "695c3049ad94fa38b7f1e8243cdcee27ecad0867"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "8.0.1000+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "7a1a306b72cfa60634f03a911405f4e64d1b718b"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.0"

[[RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "d12e612bba40d189cead6ff857ddb67bd2e6a387"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMD]]
git-tree-sha1 = "bc12e315740f3a36a6db85fa2c0212a848bd239e"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.2"

[[ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "2436b15f376005e8790e318329560dcc67188e84"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.3"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "4e051b85454b4e4f66e6a6b7bdc452ad9da3dcf6"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.10"

[[StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "13237798b407150a6d2e2bce5d793d7d9576e99e"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.13"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "f8cd5b95aae14d3d88da725414bdde342457366f"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.2"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f2fd3f288dfc6f507b0c3a2eb3bac009251e548b"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.22"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "77fea79baa5b22aeda896a8d9c6445a74500a2c2"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.74"

[[Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[WAV]]
deps = ["Base64", "FileIO", "Libdl", "Logging"]
git-tree-sha1 = "7e7e1b4686995aaf4ecaaf52f6cd824fa6bd6aa5"
uuid = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"
version = "1.2.0"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "ef4f23ffde3ee95114b461dc667ea4e6906874b2"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.0"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "66cc604b9a27a660e25a54e408b4371123a186a6"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.49"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─3037a996-6acc-11ed-2c2f-8f490121dc68
# ╠═a08f9487-6a91-4a80-a479-4fd3a7e4d0c7
# ╠═b2eef973-24f1-47c5-9ed6-5b0ad5ee3442
# ╟─6c87265a-feb9-4e3c-9076-535f81b3fc5a
# ╠═2d5ef07d-9890-431b-a8fc-0f62bd6c5f29
# ╠═687d6db7-4fc8-4425-8b33-2eaabd985a4d
# ╠═1e065d4d-beb8-49f8-972a-b39bfc2bf340
# ╠═c38e7c4e-e6aa-4f4b-9954-2cb1bcd2059d
# ╠═c438523e-8aab-4e47-8cb6-abe589ae887d
# ╠═72c3fa6b-8595-4d99-b923-9dc8d614b0d3
# ╠═b08b7f4b-925d-4e7b-8d40-ef15dc8da448
# ╠═3457dc29-a4f1-4595-bea5-120f76ac2b17
# ╟─99c054ad-5bc0-4aec-93d4-d26c20cd1b10
# ╠═5f5858eb-bf40-4b54-9512-5719e2a3cfd5
# ╠═1a441922-9f28-4628-9070-ac3a5d755790
# ╠═051ab7d8-2075-4f40-a4b8-e27fba63b80e
# ╠═1e974e18-76ad-49ee-bf23-f3078402021b
# ╠═8bed4018-de19-4266-af39-d70f8215fbc9
# ╠═19a42016-d911-4fca-b5eb-abb802740728
# ╟─f5c3d15e-0110-4c1e-a9b0-c45a6192efc2
# ╠═b756ab36-1178-4748-b10c-62fc6b6293db
# ╠═70fa026a-7c07-417e-902d-7d890762973f
# ╠═25733f1b-7ec7-4786-91a6-77dc4c1efd24
# ╠═e4b07f3e-05f0-4987-bbfa-abcc2762d9b3
# ╠═51db35f2-bf8b-47f1-8727-a35f90b58db4
# ╠═09f55e73-a120-4b7d-a66c-161728a77cc3
# ╠═2efb7844-f23c-43e0-8b42-3e84788b669c
# ╠═c52286ac-abd4-4e99-bb9c-196775258816
# ╟─99ddfadd-e56d-43fd-bfd6-2ef05df939d7
# ╠═4b083413-93e4-4439-8c5f-ae38606c55b2
# ╟─39dd2002-847e-4a74-9816-20e43ab69cb6
# ╠═3347548c-a467-40e5-81d3-cc9ee70b0b4b
# ╠═6989a48d-03d2-476d-83c7-073c0bb98ab2
# ╠═ce42db49-3c2c-4e64-a52b-ad18f9844afb
# ╠═2261341f-9c8b-4f61-abff-6aa5a81fc337
# ╠═4acd8856-c99f-415e-afe9-7b950cb4d4fc
# ╠═cccc14ce-0e5d-4039-857a-69820cec2d49
# ╠═32507e28-8796-4d78-ae87-cd73fe156e08
# ╠═4318adee-2c07-4dfc-901b-c8c229c1869c
# ╠═bff95353-bbdc-4617-8be5-6f6d5f6e82ab
# ╠═70b0f14a-7c28-461d-8e6f-53051f3030ce
# ╠═5c22cbda-e999-4857-86c7-427fcacdac06
# ╠═d7af0058-924f-41af-ad21-ba910059b111
# ╟─75a60b92-8278-4d10-9bfb-dd14fabc4fc8
# ╠═b4c3040e-13b0-48ee-8652-57f058440529
# ╟─7fd7b3d7-d3bd-46e0-89b9-81b6a81c4907
# ╠═d012ed96-5112-413d-b918-df25fac3ccae
# ╠═a3b61da5-7738-416f-98cb-9a76f83af82a
# ╟─9c7a5f70-5897-4867-839d-dbad86773b85
# ╠═ef21c525-ba3f-43a5-a0f6-79f83618a952
# ╠═9ea2cc9c-a6e3-45f7-b2b2-7823af10cf85
# ╠═16ae8d32-d54b-441b-b544-745a2c5ea314
# ╠═c011a460-2276-4704-ae65-fd7098f6e21b
# ╠═ce4b7434-f0b1-42b2-9ac6-7bf437a3f9f0
# ╟─ca047acc-c51f-4975-977f-eda50b12822c
# ╠═160deb3d-7673-46cf-bda2-cb51f51ba12d
# ╠═78049b33-4270-46e5-a5d1-9601582be47c
# ╠═b9de6ffc-b99a-403b-9c38-f2d53b611048
# ╠═656f91e1-8335-45a5-a8b2-12725888c7e1
# ╠═0f8aad0b-a1ba-4e59-9d71-5b3be612c500
# ╠═e38d4066-b520-48bb-b238-900d658eccab
# ╠═e47e9820-09ca-4b46-9f98-f8d38f0e3668
# ╠═4956e408-4b48-438f-b785-1cd67b66708b
# ╠═a8d73067-e78c-4a4d-a7a1-1324275ab192
# ╠═7f14c560-9a9e-4588-9dbc-352929569d95
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
