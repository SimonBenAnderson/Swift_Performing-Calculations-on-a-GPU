//
//  MetalAdder.swift
//  PerformingCalculationsOnAGPU_Swift
//
//  Created by Simon Anderson on 5/07/23.
//

import Foundation
import Metal

// The number of floats in each array, and the size of the arrays in bytes.
let arrayLength:Int = 1 << 24
let bufferSize:Int = arrayLength * MemoryLayout<Float>.stride

class MetalAdder {
    
    var _mDevice : MTLDevice!

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    var _mAddFunctionPSO: MTLComputePipelineState!

    // The command queue used to pass commands to the device.
    var _mCommandQueue: MTLCommandQueue!

    // Buffers to hold data.
    var _mBufferA: MTLBuffer!
    var _mBufferB: MTLBuffer!
    var _mBufferResult: MTLBuffer!
    
    init(withDevice:MTLDevice) {
        _mDevice = withDevice
        
        let defaultLibrary = _mDevice.makeDefaultLibrary()!
        
        let addFunction = defaultLibrary.makeFunction(name: "add_arrays")!
        
        do {
            _mAddFunctionPSO = try _mDevice.makeComputePipelineState(function: addFunction)
        }
        catch let errors as NSError {
            fatalError("Error: \(errors.localizedDescription)")
        }
        
        
        
        _mCommandQueue = _mDevice.makeCommandQueue()!
    }
    
    func prepareData() {
        _mBufferA = _mDevice.makeBuffer(length: bufferSize, options: .storageModeShared)
        _mBufferB = _mDevice.makeBuffer(length: bufferSize, options: .storageModeShared)
        _mBufferResult = _mDevice.makeBuffer(length: bufferSize, options: .storageModeShared)
        
        generateRandomFloatData(_mBufferA)
        generateRandomFloatData(_mBufferB)
    }
    
    func sendComputeCommand() {
        // Create a command buffer to hold commands.
        let commandBuffer = _mCommandQueue.makeCommandBuffer()!
        
        // Start a compute pass.
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        self.encodeAddCommand(computeEncoder)
        
        // End the compute pass.
        computeEncoder.endEncoding()
        
        // Execute the command.
        commandBuffer.commit()
        
        // Normally, you want to do other work in your app while the GPU is running,
        // but in this example, the code simply blocks until the calculation is complete.
        commandBuffer.waitUntilCompleted()
        
        self.verifyResults()
    }
    
    func encodeAddCommand(_ computeEncoder:MTLComputeCommandEncoder) {
        computeEncoder.setComputePipelineState(_mAddFunctionPSO)
        computeEncoder.setBuffer(_mBufferA, offset: 0, index: 0)
        computeEncoder.setBuffer(_mBufferB, offset: 0, index: 1)
        computeEncoder.setBuffer(_mBufferResult, offset: 0, index: 2)
        
        let gridSize:MTLSize = MTLSizeMake(arrayLength, 1, 1)
        
        // Calculate a threadgroup size.
        var threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup
        if threadGroupSize > arrayLength {
            threadGroupSize = arrayLength
        }
        let threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1)
        
        // Encode the compute command.
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
    
    func generateRandomFloatData(_ buffer:MTLBuffer) {
        
        let dataPtr = buffer.contents().bindMemory(to: Float.self, capacity: arrayLength)

        for index in 0..<arrayLength {
            dataPtr[index] = Float.random(in:0..<MAXFLOAT)/MAXFLOAT
        }
    }
    
    
    func verifyResults() {
        
        let a = self._mBufferA.contents().assumingMemoryBound(to: Float.self)
        let b = self._mBufferB.contents().assumingMemoryBound(to: Float.self)
        let result = self._mBufferResult.contents().assumingMemoryBound(to: Float.self)

        for index in 0..<arrayLength {
            if (result[index] != (a[index] + b[index]))
            {
                print("Compute ERROR: index=\(index) result=\(result[index]) vs \(a[index] + b[index])=a+b")
                assert(result[index] == (a[index] + b[index]))
            }
        }
        print("Compute results as expected")
        
    }
}
