//
//  main.swift
//  Custom-differentiation
//
//  Created by User on 26/09/2020.
//

import Foundation
import TensorFlow
import PythonKit
import Darwin.C

/*
print(Python.version)
var x = Tensor<Float>([[1, 2], [3, 10]])
print(x + x)
print("blob")
*/

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

print("exp(3) =", sillyExp(3))
print("𝛁exp(3) =", gradient(of: sillyExp)(3))



