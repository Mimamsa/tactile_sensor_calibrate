import serial
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
import math
import cv2


OUTPUT_INIT_FILE = 'samples/init_insole1.npy'
USE_LOG = True

PRESSURE_MAX = 600
PRESSURE_MIN = 550



def getUnixTimestamp():
    return np.datetime64(datetime.datetime.now()).astype(np.int64) / 1e6  # unix TS in secs and microsecs


def read_pressure(ser, w=32, h=32):
    """Read pressure map """
    # Request readout
    ser.reset_input_buffer() # Remove the confirmation 'w' sent by the sensor
    ser.write('a'.encode('utf-8')) # Request data from the sensor

    # Receive data
    length = 2 * w * h
    input_string = ser.read(length)
    x = np.frombuffer(input_string, dtype=np.uint8).astype(np.uint16)
    if not len(input_string) == length:
        print("Only got %d values => Drop frame." % len(input_string))
        return None

    x = x[0::2] * 32 + x[1::2]
    x = x.reshape(h, w).transpose(1, 0)
    return x


def remap(pressure):
    """Remap by swapping columns and rows of odd and even index."""
    for i in range(16):
        pressure[:,[2*i, 2*i+1]] = pressure[:, [2*i+1, 2*i]]
    for i in range(16):
        pressure[[2*i, 2*i+1],:] = pressure[[2*i+1, 2*i],:]
    return pressure


def read_data(ser):
    """Read timestamp and pressure map """
    pressure = read_pressure(ser)
    ts = getUnixTimestamp()

    if pressure is None:
            return None, 0

    ## postprocess
    pressure = remap(pressure)
    # pressure = mask(pressure)

    return pressure, ts


def letterBoxImage(img, size, return_bbox=False):
    # letter box
    szIn = np.array([img.shape[1], img.shape[0]])
    x0 = (size - szIn) // 2
    x1 = x0 + szIn

    res = np.zeros([size[1], size[0], img.shape[2]], img.dtype)
    res[x0[1]:x1[1],x0[0]:x1[0],:] = img

    if return_bbox:
        return res, np.concatenate((x0,x1-x0))

    return res


def fitImageToBounds(img, bounds, upscale=False, interpolation=cv2.INTER_LINEAR):
    inAsp = img.shape[1] / img.shape[0]
    outAsp = bounds[0] / bounds[1]

    if not upscale and img.shape[1] <= bounds[0] and img.shape[0] <= bounds[1]:
        return img
    elif img.shape[1] == bounds[0] and img.shape[0] == bounds[1]:
        return img

    if inAsp < outAsp:
        # Narrow to wide
        height = bounds[1]
        width = math.floor(inAsp * height+ 0.5)
    else:
        width = bounds[0]
        height = math.floor(width / inAsp + 0.5)

    res = cv2.resize(img, (int(width), int(height)), interpolation = interpolation)
    if len(res.shape) < len(img.shape):
        res = res[..., np.newaxis]
    return res


def resizeImageLetterBox(img, size, interpolation = cv2.INTER_LINEAR, return_bbox = False):
    img = fitImageToBounds(img, size, upscale = True, interpolation = interpolation)
    return letterBoxImage(img, size, return_bbox)


def showVizImage(im, viz_w=480, viz_h=480):
    """
    """
    ret = True
    
    if viz_w > 0 and viz_h > 0:
        im = resizeImageLetterBox(im, [viz_w, viz_h])
    cv2.imshow('noname', im)

    if cv2.waitKey(1) & 0xff == 27:  # esc
        print('Detected user termination command.')
        # self.running.value = False
        ret = False

    return ret


def render(pressure, resolution=(480,480)):
    '''
    Renders the content, returns image.
    '''
    pressure = (pressure.astype(np.float32) - PRESSURE_MIN) / (PRESSURE_MAX - PRESSURE_MIN)
    pressure = np.clip(pressure, 0, 1)
    if USE_LOG:
        pressure = np.log(pressure + 1) / np.log(2.0)

    im = cv2.applyColorMap((np.clip(pressure, 0, 1) * 255).astype('uint8'), cv2.COLORMAP_JET)

    im = fitImageToBounds(im, resolution, upscale=True, interpolation = cv2.INTER_NEAREST)
    # caption = '[%s] %06d (%.3f s)|Range=%03d(%03d)-%03d(%03d)' % (
        # topic, frame, ts, data['pressure'].min(), PRESSURE_MIN, data['pressure'].max(), PRESSURE_MAX)
    # cv2.putText(im, caption, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness = 1, lineType = cv2.LINE_AA)
    # im = image_tools.resizeImageLetterBox(im, self.resolution, interpolation = cv2.INTER_NEAREST)
    return im


def viz(ts, data):
    """
    """
    im_cur = render(data)
    ret = showVizImage(im_cur)
    return ret


def main():

    ser = serial.Serial('COM9', baudrate=500000, timeout=1.0)
    assert ser.is_open, 'Failed to open COM port!'
    
    limit = 50
    frame_count = 0
    pressure_maps = []
    
    running = True
    while(running):
        pressure, ts = read_data(ser)

        if pressure is not None:
            running = viz(ts, pressure)
            
            if frame_count<limit:
                pressure_maps.append(pressure)
                frame_count += 1
            else:
                print('Frame count reached the limit.')

    ser.close()
    
    # Find mean map
    stacked_map = np.stack(pressure_maps)
    mean_map = np.mean(stacked_map, axis=0)

    np.save(OUTPUT_INIT_FILE, mean_map)
    
    # running = True
    # while(running):
        # running = viz(ts, mean_map)


if __name__=='__main__':
    main()