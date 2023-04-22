#--------------------------------------------------------------------------------------------------------------------
# oakd_camera_stereo
# A sample python program that displays depth image from Oak-D stereo camera.
#
# Camera Specs
# Robotics Vision Core 2 (RVC2) with 16x SHAVE cores
#  -> Streaming Hybrid Architecture Vector Engine (SHAVE)
# Color camera sensor = 12MP (4032x3040 via ISP stream)
# Depth perception: baseline of 7.5cm
#  -> Ideal range: 70cm - 8m
#  -> MinZ: ~20cm (400P, extended), ~35cm (400P OR 800P, extended), ~70cm (800P)
#  -> MaxZ: ~15 meters with a variance of 10% (depth accuracy evaluation)
# https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1098OAK.html
#
# Code
# The code in this file is based on the code from Luxonis Tutorials and Code Samples.
# https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/
# https://docs.luxonis.com/projects/api/en/latest/tutorials/code_samples/
# https://github.com/luxonis/depthai-python/tree/main/examples/AprilTag
#
# Additional Info
# This website provides a good overview of the camera and how to use the NN pipeline.
# https://pyimagesearch.com/2022/12/19/oak-d-understanding-and-running-neural-network-inference-with-depthai-api/
#
# AprilTags
# https://april.eecs.umich.edu/software/apriltag
# https://github.com/AprilRobotics/apriltag-imgs
#
# Printable pdf file of AprilTags located here
# https://docs.cbteeple.com/robot/april-tags

#--------------------------------------------------------------------------------------------------------------------
#import numpy as np  # numpy package -> manipulate the packet data returned by depthai
import cv2  # opencv-python  package -> display the video stream
import depthai as dai  # depthai package -> access the camera and its data packets
import time


#--------------------------------------------------------------------------------------------------------------------
# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
aprilTag = pipeline.create(dai.node.AprilTag)
manip = pipeline.create(dai.node.ImageManip)

xoutAprilTag = pipeline.create(dai.node.XLinkOut)
xoutAprilTagImage = pipeline.create(dai.node.XLinkOut)

xoutAprilTag.setStreamName("aprilTagData")
xoutAprilTagImage.setStreamName("aprilTagImage")


#--------------------------------------------------------------------------------------------------------------------
# Properties
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

manip.initialConfig.setResize(480, 270)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)

aprilTag.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_36H11)


#--------------------------------------------------------------------------------------------------------------------
# Linking
aprilTag.passthroughInputImage.link(xoutAprilTagImage.input)
camRgb.video.link(manip.inputImage)
manip.out.link(aprilTag.inputImage)
aprilTag.out.link(xoutAprilTag.input)

# always take the latest frame as apriltag detections are slow
aprilTag.inputImage.setBlocking(False)
aprilTag.inputImage.setQueueSize(1)


#--------------------------------------------------------------------------------------------------------------------
# advanced settings, configurable at runtime
# aprilTagConfig = aprilTag.initialConfig.get()
# aprilTagConfig.quadDecimate = 4
# aprilTagConfig.quadSigma = 0
# aprilTagConfig.refineEdges = True
# aprilTagConfig.decodeSharpening = 0.25
# aprilTagConfig.maxHammingDistance = 1
# aprilTagConfig.quadThresholds.minClusterPixels = 5
# aprilTagConfig.quadThresholds.maxNmaxima = 10
# aprilTagConfig.quadThresholds.criticalDegree = 10
# aprilTagConfig.quadThresholds.maxLineFitMse = 10
# aprilTagConfig.quadThresholds.minWhiteBlackDiff = 5
# aprilTagConfig.quadThresholds.deglitch = False
# aprilTag.initialConfig.set(aprilTagConfig)


#--------------------------------------------------------------------------------------------------------------------
# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the mono frames from the outputs defined above
    manipQueue = device.getOutputQueue("aprilTagImage", 8, False)
    aprilTagQueue = device.getOutputQueue("aprilTagData", 8, False)

    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while(True):
        inFrame = manipQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        monoFrame = inFrame.getFrame()
        frame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)

        aprilTagData = aprilTagQueue.get().aprilTags
        for aprilTag in aprilTagData:
            topLeft = aprilTag.topLeft
            topRight = aprilTag.topRight
            bottomRight = aprilTag.bottomRight
            bottomLeft = aprilTag.bottomLeft

            center = (int((topLeft.x + bottomRight.x) / 2), int((topLeft.y + bottomRight.y) / 2))

            cv2.line(frame, (int(topLeft.x), int(topLeft.y)), (int(topRight.x), int(topRight.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(topRight.x), int(topRight.y)), (int(bottomRight.x), int(bottomRight.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(bottomRight.x), int(bottomRight.y)), (int(bottomLeft.x), int(bottomLeft.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(bottomLeft.x), int(bottomLeft.y)), (int(topLeft.x), int(topLeft.y)), color, 2, cv2.LINE_AA, 0)

            idStr = "ID: " + str(aprilTag.id)
            cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.putText(frame, "Fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

        cv2.imshow("April tag frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break