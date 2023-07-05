//
//  main.swift
//  PerformingCalculationsOnAGPU_Swift
//
//  Created by Simon Anderson on 5/07/23.
//

import Foundation
import Metal

func main() {
    
    // using MTLCopyAllDevices instead of MTLCreateSystemDefaultDevice, as this is a terminal command.
    let device = MTLCopyAllDevices()[0]
        
    // Create the custom object used to encapsulate the Metal code.
    // Initializes objects to communicate with the GPU.
    let adder = MetalAdder(withDevice: device)
    
    // Create buffers to hold data
    adder.prepareData()
    
    // Send a command to the GPU to perform the calculation.
    adder.sendComputeCommand()
    
    print("Execution finished")
}

main()
