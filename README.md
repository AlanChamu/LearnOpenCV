# Learn OpenCV


## Before
![Lane Before](src/test2.JPG)

## After
![Lane_After](src/test2wlanes.jpg)

### Dependencies

#### Core
```
pip install cv2     # computer vision library
pip install numpy   # support for large, multi-dimensional arrays
```

#### Additional
```
pip install pyserial    # communication with arduino
pip install matplotlib  # to get numerical value for coordinates
pip install cvlib       # for object detection
pip install tensorflow  # required for cvlib
```

<!-- diagram of how the auto rc will work  -->

# Units
* input unit        (part of this repo - tcp server to get live footage)
(rceyes.py?)
* processing unit   (what repo will focus on)
(rccortex.py)
* rc control unit   (part of this repo - pyserial to send signals to arduino)
(rcbellum.py)
### Sources
https://www.youtube.com/watch?v=kft1AJ9WVDk
