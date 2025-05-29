def process_boxes(box):
    xmin = box.left()
    ymin = box.top()
    xmax = box.right()
    ymax = box.bottom()
    return [int(xmin), int(ymin), int(xmax), int(ymax)]