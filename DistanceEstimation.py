def SWAB():
    import cv2 as cv
    import numpy as np

    # Distance constants in inches 
    KNOWN_DISTANCE = 45

    # Width constants in inches
    PERSON_WIDTH = 16 
    MOBILE_WIDTH = 3.0
    BOTTLE_WIDTH = 3.0
    BOOK_WIDTH=4.5
    STOPSIGN_WIDTH=30
    FIREHYDRANT_WIDTH=4.5
    VASE_WIDTH=9
    CLOCK_WIDTH=10
    TRAFFICLIGHT_WIDTH=9.5
    KEYBOARD_WIDTH=17
    REMOTE_WIDTH=2
    MOUSE_WIDTH=5
    LAPTOP_WIDTH=14
    BED_WIDTH=36
    POTTEDPLANT_WIDTH=12

    # Object detector constant 
    CONFIDENCE_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.3

    # colors for object detected
    COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    GREEN =(0,255,0)
    BLACK =(0,0,0)
    # defining fonts 
    FONTS = cv.FONT_HERSHEY_COMPLEX

    # getting class names from classes.txt file 
    class_names = []
    with open("classes.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    #  setttng up opencv net
    yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

    yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    model = cv.dnn_DetectionModel(yoloNet)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    # object detector funciton /method
    def object_detector(image):
        classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        # creating empty list to add objects data
        data_list =[]
        for (classid, score, box) in zip(classes, scores, boxes):
            # define color of each, object based on its class id 
            color= COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid], score)
            
            # draw rectangle on and label on object
            cv.rectangle(image, box, color, 2)
            cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            
            # getting the data 
            # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)

            if classid ==0: # person class id 
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==67: # cellphone
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==39: # bottle
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==11: # stop-sign
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==10: # fire-hydrant
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==75: # flower vase
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==74: # clock
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==9: # traffic light
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==66: # keyboard
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==65: # remote
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==64: # mouse
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==63: # laptop
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==59: # bed
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            elif classid ==58: # potted plant
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])



            # elif classid ==73: # Book
            #     data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

            # if you want inclulde more classes then you have to simply add more [elif] statements here
            # returning list containing the object data. 
        return data_list

    def focal_length_finder (measured_distance, real_width, width_in_rf):
        focal_length = (width_in_rf * measured_distance) / real_width

        return focal_length

    # distance finder function 
    def distance_finder(focal_length, real_object_width, width_in_frmae):
        distance = (real_object_width * focal_length) / width_in_frmae
        return distance

    # reading the reference image from dir 
    ref_person = cv.imread('ReferenceImages/image14.png')
    ref_mobile = cv.imread('ReferenceImages/image4.png')
    ref_bottle = cv.imread('ReferenceImages/image2.png')
    ref_stopsign = cv.imread('ReferenceImages/image18.jpg')
    ref_firehydrant = cv.imread('ReferenceImages/image19.jpg')
    ref_vase = cv.imread('ReferenceImages/image20.jpg')
    ref_clock = cv.imread('ReferenceImages/image21.jpg')
    ref_trafficlight = cv.imread('ReferenceImages/image22.jpg')
    ref_keyboard = cv.imread('ReferenceImages/image23.png')
    ref_remote = cv.imread('ReferenceImages/image24.jpeg')
    ref_mouse = cv.imread('ReferenceImages/image25.jpg')
    ref_laptop = cv.imread('ReferenceImages/image26.jpg')
    ref_bed = cv.imread('ReferenceImages/image27.png')
    ref_pottedplant = cv.imread('ReferenceImages/image28.jpg')
    # ref_book = cv.imread('ReferenceImages/image17.jpg')

    person_data = object_detector(ref_person)
    person_width_in_rf = person_data[0][1]

    mobile_data = object_detector(ref_mobile)
    mobile_width_in_rf = mobile_data[0][1]

    bottle_data = object_detector(ref_bottle)
    bottle_width_in_rf = bottle_data[0][1]

    stopsign_data=object_detector(ref_stopsign)
    stopsign_width_in_rf=stopsign_data[0][1]

    firehydrant_data=object_detector(ref_firehydrant)
    firehydrant_width_in_rf=firehydrant_data[0][1]

    vase_data=object_detector(ref_vase)
    vase_width_in_rf=vase_data[0][1]

    clock_data=object_detector(ref_clock)
    clock_width_in_rf=clock_data[0][1]

    trafficlight_data=object_detector(ref_trafficlight)
    trafficlight_width_in_rf=trafficlight_data[0][1]

    keyboard_data=object_detector(ref_keyboard)
    keyboard_width_in_rf=keyboard_data[0][1]

    remote_data=object_detector(ref_remote)
    remote_width_in_rf=remote_data[0][1]

    mouse_data=object_detector(ref_mouse)
    mouse_width_in_rf=mouse_data[0][1]

    laptop_data=object_detector(ref_laptop)
    laptop_width_in_rf=laptop_data[0][1]

    bed_data=object_detector(ref_bed)
    bed_width_in_rf=bed_data[0][1]

    pottedplant_data=object_detector(ref_pottedplant)
    pottedplant_width_in_rf=pottedplant_data[0][1]

    # book_data = object_detector(ref_book)
    # book_width_in_rf = book_data[0][1]



    print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

    # finding focal length 
    focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

    focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

    focal_bottle = focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, bottle_width_in_rf)

    focal_stopsign = focal_length_finder(KNOWN_DISTANCE, STOPSIGN_WIDTH, stopsign_width_in_rf)

    focal_firehydrant = focal_length_finder(KNOWN_DISTANCE, FIREHYDRANT_WIDTH, firehydrant_width_in_rf)

    focal_vase = focal_length_finder(KNOWN_DISTANCE, VASE_WIDTH, vase_width_in_rf)

    focal_clock = focal_length_finder(KNOWN_DISTANCE, CLOCK_WIDTH, clock_width_in_rf)

    focal_trafficlight = focal_length_finder(KNOWN_DISTANCE, TRAFFICLIGHT_WIDTH, trafficlight_width_in_rf)

    focal_keyboard = focal_length_finder(KNOWN_DISTANCE, KEYBOARD_WIDTH, keyboard_width_in_rf)

    focal_remote = focal_length_finder(KNOWN_DISTANCE, REMOTE_WIDTH, remote_width_in_rf)

    focal_mouse = focal_length_finder(KNOWN_DISTANCE, MOUSE_WIDTH, mouse_width_in_rf)

    focal_laptop = focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, laptop_width_in_rf)

    focal_bed = focal_length_finder(KNOWN_DISTANCE, BED_WIDTH, bed_width_in_rf)

    focal_pottedplant = focal_length_finder(KNOWN_DISTANCE, POTTEDPLANT_WIDTH, pottedplant_width_in_rf)



    # focal_book = focal_length_finder(KNOWN_DISTANCE, BOOK_WIDTH, book_width_in_rf)

    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        data = object_detector(frame) 
        for i in range(1):
            for d in data:
                print(d)
                if d[0] =='person':
                    p = 'person'
                    distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
                    x, y = d[2]

                elif d[0] =='cell phone':
                    p = 'cell phone'
                    distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
                    x, y = d[2]

                elif d[0] =='bottle':
                    p = 'bottle'
                    distance = distance_finder (focal_bottle, BOTTLE_WIDTH, d[1])
                    x, y = d[2]

                elif d[0] =='fire-hydrant':
                    p = 'fire-hydrant'
                    distance = distance_finder (focal_firehydrant, FIREHYDRANT_WIDTH, d[1])
                    x, y = d[2]

                elif d[0] =='keyboard':
                    p = 'keyboard'
                    distance = distance_finder (focal_keyboard, KEYBOARD_WIDTH, d[1])
                    x, y = d[2]
                
                elif d[0] =='trafficlight':
                    p = 'trafficlight'
                    distance = distance_finder (focal_trafficlight, TRAFFICLIGHT_WIDTH, d[1])
                    x, y = d[2]
                    
                elif d[0] =='remote':
                    p = 'remote'
                    distance = distance_finder (focal_remote, REMOTE_WIDTH, d[1])
                    x, y = d[2]
                    
                elif d[0] =='mouse':
                    p = 'mouse'
                    distance = distance_finder (focal_mouse, MOUSE_WIDTH, d[1])
                    x, y = d[2]
                    
                elif d[0] =='vase':
                    p = 'vase'
                    distance = distance_finder (focal_vase, VASE_WIDTH, d[1])
                    x, y = d[2]
                    
                elif d[0] =='clock':
                    p = 'clock'
                    distance = distance_finder (focal_clock, CLOCK_WIDTH, d[1])
                    x, y = d[2]
                    
                elif d[0] =='laptop':
                    p = 'laptop'
                    distance = distance_finder (focal_laptop, LAPTOP_WIDTH, d[1])
                    x, y = d[2]
                    
                elif d[0] =='bed':
                    p = 'bed'
                    distance = distance_finder (focal_bed, BED_WIDTH, d[1])
                    x, y = d[2]
                    
                elif d[0] =='pottedplant':
                    p = 'pottedplant'
                    distance = distance_finder (focal_pottedplant, POTTEDPLANT_WIDTH, d[1])
                    x, y = d[2]

                # elif d[0] =='book':
                #     p = 'book'
                #     distance = distance_finder (focal_book, BOOK_WIDTH, d[1])
                #     x, y = d[2]
                    
                cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
                cv.putText(frame, f'Dis: {round(distance,2)} ', (x+5,y+13), FONTS, 0.48, GREEN, 2)
                output= p + "infront of you in" + str(round(distance)) + "inches"
                
                import speech_recognition as sr
                import pyttsx3
                abc=pyttsx3.init()
                speech_converting_sentence=output
                voice=abc.getProperty('voices')
                abc.setProperty('rate',120)
                abc.setProperty('volume',2.0)
                abc.setProperty('voice',voice[1].id)
                abc.say("there is a "+speech_converting_sentence)
                abc.runAndWait()   
                    
                    
            
        cv.imshow('frame',frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    cap.release()

