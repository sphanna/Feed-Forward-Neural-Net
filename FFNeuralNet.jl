using DualNumbers

mutable struct Network
    Layers
    Weights
    Biases
    function Network(layers,weights,biases)
        new(deepcopy(layers),deepcopy(weights),deepcopy(biases))
    end
end

function getError(NN,output)
    e = NN.Layers[end] - output
    return 0.5*(e'e)
end

function forward!(NN,input,f;bias = true)
    NN.Layers[1] = input;
    W = NN.Weights; L = NN.Layers; B = NN.Biases; n = length(W);
    for i in 1:n
        Activation = W[i]*L[i] + (bias && B[i])
        NN.Layers[i+1] = f.(Activation)
    end
    return NN
end

function backward!(NN,out,fp;bias=true)
    W = NN.Weights; L = NN.Layers; B = NN.Biases; n = length(W);
    δ = Vector{Any}(undef, n)
    δ[end] = (L[end] .- out).*(fp.(W[end]*L[end-1] + (bias && B[end])))
    for i in n-1:1
        δ[i] = (W[i+1]'δ[i+1]).*fp.(W[i]*L[i] + (bias && B[i]))  
    end
    return δ
end

function update!(NN,δ,η;bias=true)
    W = NN.Weights; L = NN.Layers; B = NN.Biases; n = length(W);
    for i in 1:n     
        W[i] .= W[i] .- η*δ[i]*L[i]'  
    end
    bias && (for i in 1:n  B[i] = B[i] - η*δ[i]  end)
    return NN
end

function forwardDual!(NN,input,f;bias = true)
    NN.Layers[1] = input;
    W = NN.Weights; L = NN.Layers; B = NN.Biases; n = length(W);
    for i in 1:n
        Activation = W[i]*realpart.(L[i]) + (bias && B[i])
        NN.Layers[i+1] = f.(Dual.(Activation,1.0))
    end
    return NN
end

function backwardDual!(NN,out;bias=true)
    W = NN.Weights; L = NN.Layers; B = NN.Biases; n = length(W);
    δ = Vector{Any}(undef, n)
    δ[end] = (realpart.(L[end]) .- out).*dualpart.(L[end])
    for i in n:2
        δ[i-1] = (W[i]'δ[i]).*dualpart.(L[i])
    end
    return δ
end

function updateDual!(NN,δ,η;bias=true)
    W = NN.Weights; L = NN.Layers; B = NN.Biases; n = length(W);
    for i in 1:n     
        W[i] .= W[i] .- η*δ[i]*realpart.(L[i])'  
    end
    bias && (for i in 1:n  B[i] = B[i] - η*δ[i]  end)
    return NN
end

function FWBP!(NN,input,output,f,fp,η;bias = true)
    forward!(NN,input,f,bias = bias)
    δ = backward!(NN,output,fp,bias = bias)
    update!(NN,δ,η,bias = bias)
end

function FWBPDual!(NN,input,output,f,η;bias = true)
    forwardDual!(NN,input,f,bias = bias)
    δ = backwardDual!(NN,output,bias = bias)
    update!(NN,δ,η,bias = bias)
end

function FWBPonline!(NN,input,output,f,fp,η,N;bias=true)
    for k in 1:N
        for i in 1:length(input)
            FWBP!(NN,input[i],output[i],f,fp,η,bias = bias)
        end
    end
    return NN
end

function FWBPbatch!(NN,input,output,f,fp,η,N;bias=true)
    for i in 1:length(input)
        for k in 1:N
            FWBP!(NN,input[i],output[i],f,fp,η,bias = bias)
        end
    end
    return NN
end

function FWBPbatchTest!(NN,input,output,f,fp,η,N;bias=true)
    for k in 1:N
        forward!(NN,input[1],f)
        dW = backward!(NN,output[1],fp)
        if(length(input) > 1)
            for i in 2:length(input)
                forward!(NN,input[i],f)
                dW = dW .+ backward!(NN,output[i],fp)
            end
        end
        update!(NN,dW,η,bias = bias)
    end
    return NN
end

sigmoid(x) = 1 / (1 + exp(-x))
sigmoidp(x) = sigmoid(x)*(1-sigmoid(x))

begin #setup
    η = 1.0
    input = [[1.0,1.0],[-1.0,-1.0]]; output = [[0.9],[0.05]]
    W1 = [0.3 0.3; 0.3 0.3];  WL = [0.8 0.8]
    bias = 0.0; B1 = [bias, bias]; BL = [bias]
    WeightsInit = [W1,WL]
    BiasesInit = [B1,BL]
    nLayers = length(WeightsInit)+1
    LayersInit = Vector{Any}(undef, nLayers)
end

#method 1
NN = Network(LayersInit,WeightsInit,BiasesInit)
FWBPonline!(NN,input,output,sigmoid,sigmoidp,η,15,bias=true)

forward!(NN,input[1],sigmoid)
getError(NN,output[1])

forward!(NN,input[2],sigmoid)
getError(NN,output[2])

#method 2
NN = Network(LayersInit,WeightsInit,BiasesInit)
FWBPbatch!(NN,input,output,sigmoid,sigmoidp,η,15,bias=true)

forward!(NN,input[1],sigmoid)
getError(NN,output[1])

forward!(NN,input[2],sigmoid)
getError(NN,output[2])

