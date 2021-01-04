import Cocoa
import MetalPerformanceShaders

var bufA: [Float] = [1, 2, 3, 4]
var bufB: [Float] = [1, 2, 3, 4]
var bufC: [Float] = [1, 1, 1, 1]

let ptrA = UnsafeMutablePointer<Float>(mutating: bufA)
let ptrB = UnsafeMutablePointer<Float>(mutating: bufB)
let ptrC = UnsafeMutablePointer<Float>(mutating: bufC)

let iDev = MTLCreateSystemDefaultDevice()!
let iQueue = iDev.makeCommandQueue()!
let iCmd = iQueue.makeCommandBuffer()!

let datA = iDev.makeBuffer(bytes: ptrA,
                           length: 4 * MemoryLayout<Float>.size)
let datB = iDev.makeBuffer(bytes: ptrB,
                           length: 4 * MemoryLayout<Float>.size)
let datC = iDev.makeBuffer(bytes/*NoCopy*/: ptrC,
                           length: 4 * MemoryLayout<Float>.size,
                           options: .storageModeShared)

let mA = MPSMatrix(buffer: datA!,
                   descriptor: MPSMatrixDescriptor(rows: 2, columns: 2,
                                                   rowBytes: 2 * MemoryLayout<Float>.size,
                                                   dataType: .float32))
let mB = MPSMatrix(buffer: datB!,
                   descriptor: MPSMatrixDescriptor(rows: 2, columns: 2,
                                                   rowBytes: 2 * MemoryLayout<Float>.size,
                                                   dataType: .float32))
let mC = MPSMatrix(buffer: datC!,
                   descriptor: MPSMatrixDescriptor(rows: 2, columns: 2,
                                                   rowBytes: 2 * MemoryLayout<Float>.size,
                                                   dataType: .float32))

let sGemm = MPSMatrixMultiplication(device: iDev,
                                    transposeLeft: false,
                                    transposeRight: false,
                                    resultRows: 2,
                                    resultColumns: 2,
                                    interiorColumns: 2,
                                    alpha: 1.0, beta: 1.0)

sGemm.encode(commandBuffer: iCmd,
             leftMatrix: mA,
             rightMatrix: mB,
             resultMatrix: mC)
iCmd.commit()
iCmd.waitUntilCompleted()

let resC = datC!.contents().bindMemory(to: Float32.self, capacity: 1)
let lstC = UnsafeBufferPointer(start: resC, count: 4)
print(Array(lstC))
