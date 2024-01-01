from fusion import fuse_edge_cloud_detections
from model import Detection, DetectionType

current1 = [Detection(category='people', score=52, bbox=[640, 710, 15, 26]),
            Detection(category='car', score=92, bbox=[706, 721, 37, 70]),
            Detection(category='car', score=87, bbox=[785, 471, 26, 26]),
            Detection(category='car', score=87, bbox=[637, 632, 34, 54]),
            Detection(category='car', score=84, bbox=[773, 429, 22, 22]),
            Detection(category='car', score=84, bbox=[693, 488, 26, 28]),
            Detection(category='car', score=82, bbox=[689, 533, 28, 37]),
            Detection(category='car', score=78, bbox=[704, 357, 15, 14]),
            Detection(category='car', score=73, bbox=[885, 531, 37, 36]),
            Detection(category='car', score=67, bbox=[621, 399, 19, 21]),
            Detection(category='car', score=67, bbox=[660, 440, 22, 24]),
            Detection(category='car', score=66, bbox=[721, 443, 20, 25]),
            Detection(category='car', score=60, bbox=[1158, 460, 29, 27]),
            Detection(category='car', score=57, bbox=[720, 292, 10, 10]),
            Detection(category='car', score=55, bbox=[649, 538, 29, 36]),
            Detection(category='car', score=52, bbox=[826, 452, 17, 20]),
            Detection(category='motor', score=61, bbox=[614, 757, 17, 32])]
new1 = [Detection(category='people', score=37, bbox=[777, 666, 12, 20]),
        Detection(category='car', score=99, bbox=[1188, 728, 90, 72]),
        Detection(category='car', score=99, bbox=[1154, 469, 25, 27]),
        Detection(category='car', score=99, bbox=[1079, 465, 36, 20]),
        Detection(category='car', score=99, bbox=[1230, 451, 45, 24]),
        Detection(category='car', score=99, bbox=[1265, 458, 45, 27]),
        Detection(category='car', score=99, bbox=[695, 623, 36, 72]),
        Detection(category='car', score=99, bbox=[691, 563, 30, 40]),
        Detection(category='car', score=36, bbox=[779, 339, 24, 13]),
        Detection(category='awning-tricycle', score=31, bbox=[826, 470, 18, 23]),
        Detection(category='motor', score=32, bbox=[445, 529, 19, 17])]

result1 = fuse_edge_cloud_detections(current1, new1, DetectionType.CLOUD)

current2 = [Detection(category='people', score=37, bbox=[777, 666, 12, 20]),
            Detection(category='car', score=99, bbox=[1188, 728, 90, 72]),
            Detection(category='car', score=99, bbox=[1154, 469, 25, 27]),
            Detection(category='car', score=99, bbox=[1079, 465, 36, 20]),
            Detection(category='car', score=99, bbox=[1230, 451, 45, 24]),
            Detection(category='car', score=99, bbox=[1265, 458, 45, 27]),
            Detection(category='car', score=99, bbox=[695, 623, 36, 72]),
            Detection(category='car', score=99, bbox=[691, 563, 30, 40]),
            Detection(category='car', score=36, bbox=[779, 339, 24, 13]),
            Detection(category='awning-tricycle', score=31, bbox=[826, 470, 18, 23]),
            Detection(category='motor', score=32, bbox=[445, 529, 19, 17])]
new2 = [Detection(category='car', score=91, bbox=[637, 647, 36, 59]),
        Detection(category='car', score=90, bbox=[689, 503, 28, 36]),
        Detection(category='car', score=88, bbox=[695, 542, 32, 39]),
        Detection(category='car', score=83, bbox=[708, 361, 14, 15]),
        Detection(category='car', score=81, bbox=[657, 449, 23, 25]),
        Detection(category='car', score=73, bbox=[1161, 462, 30, 26]),
        Detection(category='car', score=71, bbox=[785, 443, 25, 27]),
        Detection(category='car', score=68, bbox=[625, 400, 19, 22]),
        Detection(category='car', score=66, bbox=[720, 446, 23, 28]),
        Detection(category='car', score=65, bbox=[733, 294, 10, 10]),
        Detection(category='car', score=61, bbox=[818, 444, 24, 23]),
        Detection(category='car', score=56, bbox=[1281, 457, 35, 25]),
        Detection(category='car', score=51, bbox=[1243, 455, 37, 22]),
        Detection(category='car', score=50, bbox=[648, 549, 30, 38]),
        Detection(category='motor', score=62, bbox=[650, 732, 16, 35]),
        Detection(category='motor', score=54, bbox=[523, 599, 14, 25])]

result2 = fuse_edge_cloud_detections(current2, new2, DetectionType.EDGE)

print(current1)
print(new1)
print(result1)

print(current2)
print(new2)
print(result2)

print("Cloud:")
print("Len current1:", len(current1))
print("Len new1:", len(new1))
print("Len result1:", len(result1))
print()
print("Edge:")
print("Len current2:", len(current2))
print("Len new2:", len(new2))
print("Len result2:", len(result2))
