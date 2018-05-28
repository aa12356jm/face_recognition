'''
运行此模块可以检测对几种分辨率的图像进行基准测试


'''


import timeit

# Note: This example is only tested with Python 3 (not Python 2)

# This is a very simple benchmark to give you an idea of how fast each step of face recognition will run on your system.
# Notice that face detection gets very slow at large image sizes. So you might consider running face detection on a
# scaled down version of your image and then running face encodings on the the full size image.

#用来测试的几种不同大小的图像
TEST_IMAGES = [
    "obama-240p.jpg",
    "obama-480p.jpg",
    "obama-720p.jpg",
    "obama-1080p.jpg"
]


def run_test(setup, test, iterations_per_test=5, tests_to_run=10):
    fastest_execution = min(timeit.Timer(test, setup=setup).repeat(tests_to_run, iterations_per_test))
    execution_time = fastest_execution / iterations_per_test
    fps = 1.0 / execution_time
    return execution_time, fps

#加载图像
setup_locate_faces = """
import face_recognition

image = face_recognition.load_image_file("{}")
"""

#人脸检测
test_locate_faces = """
face_locations = face_recognition.face_locations(image)
"""

#人脸关键点检测，首先进行人脸识别，然后根据检测到的人脸区域检测关键点
setup_face_landmarks = """
import face_recognition

image = face_recognition.load_image_file("{}")
face_locations = face_recognition.face_locations(image)
"""

#检测人脸关键点
test_face_landmarks = """
landmarks = face_recognition.face_landmarks(image, face_locations=face_locations)[0]
"""

#人脸编码，首先检测人脸区域，然后对人脸区域进行抽取特征编码为128D向量
setup_encode_face = """
import face_recognition

image = face_recognition.load_image_file("{}")
face_locations = face_recognition.face_locations(image)
"""

#人脸区域编码为128维向量
test_encode_face = """
encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
"""

#直接对图像进行人脸检测，编码，端到端
setup_end_to_end = """
import face_recognition

image = face_recognition.load_image_file("{}")
"""

test_end_to_end = """
encoding = face_recognition.face_encodings(image)[0]
"""

print("Benchmarks (Note: All benchmarks are only using a single CPU core)")
print()

#对每张图像使用几种方法测试
for image in TEST_IMAGES:
    size = image.split("-")[1].split(".")[0]
    print("Timings at {}:".format(size))
    #人脸检测
    print(" - Face locations: {:.4f}s ({:.2f} fps)".format(*run_test(setup_locate_faces.format(image), test_locate_faces)))
    #关键点检测
    print(" - Face landmarks: {:.4f}s ({:.2f} fps)".format(*run_test(setup_face_landmarks.format(image), test_face_landmarks)))
    #人脸编码
    print(" - Encode face (inc. landmarks): {:.4f}s ({:.2f} fps)".format(*run_test(setup_encode_face.format(image), test_encode_face)))
    #端到端
    print(" - End-to-end: {:.4f}s ({:.2f} fps)".format(*run_test(setup_end_to_end.format(image), test_end_to_end)))
    print()
