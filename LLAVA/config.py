batch_size=2, 
shuffle=True, 
num_workers=4
num_epochs = 10
path="/home/xyc/ML/data/AniPersonaCaps/images/"
root = "/home/xyc/ML/LLAVA/"
# The first is the path to the image to be tested, and the second is the text description to be validated
test_cases = [
    ("dog.jpg","This is a very sad, and very mysterious girl, she is looking back at the camera")
]