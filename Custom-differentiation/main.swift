import Foundation
import TensorFlow
import PythonKit
import Darwin.C


print("# Create custom derivatives:")

func sillyExp(_ x: Float) -> Float {
    let 𝑒 = Float(M_E)
    print("Taking 𝑒(\(𝑒)) to the power of \(x)!")
    return pow(𝑒, x)
}

@derivative(of: sillyExp)
func sillyDerivative(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
    let y = sillyExp(x)
    return (value: y, pullback: { v in v * y })
}

print("exp(5) =", sillyExp(5))
print("𝛁exp(5) =", gradient(of: sillyExp)(5))
print("-------------------------------------------------------")

print("# stop gradient:")
/*
 method withoutDerivative(at:) stops derivatives from propagating.
 method without Derivative (When it is detectable that the derivative of a function will always be zero)
*/

let x1: Float = 8.0
let y1: Float = 21.0
var result1 = gradient(at: x1, y1) { x, y in
    sin(sin(sin(x))) + withoutDerivative(at: cos(cos(cos(y))))
}
print(result1)
print("-------------------------------------------------------")


print("# Derivative surgery:")
/*
 Method withDerivative(_:) makes arbitrary operations (including mutation) run on the gradient at a value during
 the enclosing function’s backpropagation.
*/

var x2: Float = 30
var result = gradient(at: x2) { x -> Float in
    // Print the partial derivative with respect to the result of `sin(x)`.
    let a = sin(x).withDerivative { print("∂+/∂sin = \($0)") }
    // Force the partial derivative with respect to `x` to be `0.5`.
    let b = log(x.withDerivative { (dx: inout Float) in
        print("∂log/∂x = \(dx), but rewritten to 0.5");
        dx = 0.5
    })
    return a + b
}

print(result)
print("-------------------------------------------------------")


print("# Use the data in a neural network module!")

import TensorFlow

struct MLP: Layer {
    var layer1 = Dense<Float>(inputSize: 2, outputSize: 10, activation: relu)
    var layer2 = Dense<Float>(inputSize: 10, outputSize: 1, activation: relu)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float> = [0, 1, 1, 0]) -> Tensor<Float> {
        let h0 = layer1(input).withDerivative { print("∂L/∂layer1 =", $0) }
        return layer2(h0)
    }
}

var classifier = MLP()
let optimizer = SGD(for: classifier, learningRate: 0.02)

let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
let y: Tensor<Float> = [0, 1, 1, 0]

for _ in 0..<10 {
    let 𝛁model = gradient(at: classifier) { classifier -> Tensor<Float> in
        let ŷ = classifier(x).withDerivative { print("∂L/∂ŷ =", $0) }
        let loss = (ŷ - y).squared().mean()
        print("Loss: \(loss)")
        return loss
    }
    optimizer.update(&classifier, along: 𝛁model)
}
print("-------------------------------------------------------")


print("# Recomputing activations:")
func makeRecomputedInGradient<T: Differentiable, U: Differentiable>(
    _ original: @escaping @differentiable (T) -> U
) -> @differentiable (T) -> U {
    return differentiableFunction { x in
        (value: original(x), pullback: { v in pullback(at: x, in: original)(v) })
    }
}

print("# Verify it works:")
let input: Float = 10.0
print("Running original computation...")

// Differentiable multiplication with checkpointing.
let square = makeRecomputedInGradient { (x: Float) -> Float in
    print("  Computing square...")
    return x * x
}

// Differentiate `f(x) = (cos(x))^2`.
let (output, backprop) = valueWithPullback(at: input) { input -> Float in
    return square(cos(input))
}
print("Running backpropagation...")
let grad = backprop(1)
print("Gradient = \(grad)")
print("-------------------------------------------------------")


print("Extend to neural network modules:")
struct Model: Layer {
    var conv = Conv2D<Float>(filterShape: (5, 5, 3, 6))
    var maxPool = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var flatten = Flatten<Float>()
    var dense = Dense<Float>(inputSize: 36 * 6, outputSize: 10)

    @differentiable
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv, maxPool, flatten, dense)
    }
}

input.sequenced(in: context, through: conv, maxPool, flatten, dense)

// Same as the previous `makeRecomputedInGradient(_:)`, except it's for binary functions.
func makeRecomputedInGradient<T: Differentiable, U: Differentiable, V: Differentiable>(
    _ original: @escaping @differentiable (T, U) -> V
) -> @differentiable (T, U) -> V {
    return differentiableFunction { x, y in
        (value: original(x, y), pullback: { v in pullback(at: x, y, in: original)(v) })
    }
}

/// A layer wrapper that makes the underlying layer's activations be discarded during application
/// and recomputed during backpropagation.
struct ActivationDiscarding<Wrapped: Layer>: Layer {
    /// The wrapped layer.
    var wrapped: Wrapped

    @differentiable
    func callAsFunction(_ input: Wrapped.Input) -> Wrapped.Output {
        let apply = makeRecomputedInGradient { (layer: Wrapped, input: Input) -> Wrapped.Output in
            print("    Applying \(Wrapped.self) layer...")
            return layer(input)
        }
        return apply(wrapped, input)
    }
}

extension Layer {
    func discardingActivations() -> ActivationDiscarding<Self> {
        return ActivationDiscarding(wrapped: self)
    }
}

var conv = Conv2D<Float>(filterShape: (5, 5, 3, 6)).discardingActivations()

struct Model: Layer {
    var conv = Conv2D<Float>(filterShape: (5, 5, 3, 6)).discardingActivations()
    var maxPool = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var flatten = Flatten<Float>()
    var dense = Dense<Float>(inputSize: 36 * 6, outputSize: 10)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv, maxPool, flatten, dense)
    }
}

// Use random training data.
let Tx = Tensor<Float>(randomNormal: [10, 16, 16, 3])
let Ty = Tensor<Int32>(rangeFrom: 0, to: 10, stride: 1)

var model = Model()
let opt = SGD(for: model)

for i in 1...5 {
    print("Starting training step \(i)")
    print("  Running original computation...")
    let (logits, backprop) = model.appliedForBackpropagation(to: Tx)
    let (loss, dL_dŷ) = valueWithGradient(at: logits) { logits in
        softmaxCrossEntropy(logits: logits, labels: Ty)
    }
    print("  Loss: \(loss)")
    print("  Running backpropagation...")
    let (dL_dθ, _) = backprop(dL_dŷ)
    
    opt.update(&model, along: dL_dθ)
}


