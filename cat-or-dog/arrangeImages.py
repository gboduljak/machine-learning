import os, shutil
import os.path as path

resourcesPath = path.join(os.getcwd(), 'datasets/training/all')
fullPath = path.join(os.getcwd(), 'datasets/training/full')
trainPath = path.join(os.getcwd(), 'datasets/training/train')
validationPath = path.join(os.getcwd(), 'datasets/training/validation')
testPath = path.join(os.getcwd(), 'datasets/training/test')
submissionPath = path.join(os.getcwd(), 'datasets/submission')

trainCatNames = ['cat.{}.jpg'.format(i) for i in range(1000)]
trainDogNames = ['dog.{}.jpg'.format(i) for i in range(1000)]

validationCatNames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
validationDogNames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]

testCatNames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
testDogNames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]

for trainCat in trainCatNames:
    src = path.join(resourcesPath, trainCat)
    dst = path.join(trainPath + '/cat', trainCat)
    shutil.copyfile(src, dst)

for trainDog in trainDogNames:
    src = path.join(resourcesPath, trainDog)
    dst = path.join(trainPath + '/dog', trainDog)
    shutil.copyfile(src, dst)
    
for validationCat in validationCatNames:
    src = path.join(resourcesPath, validationCat)
    dst = path.join(validationPath + '/cat', validationCat)
    shutil.copyfile(src, dst)

for validationDog in validationDogNames:
    src = path.join(resourcesPath, validationDog)
    dst = path.join(validationPath + '/dog', validationDog)
    shutil.copyfile(src, dst)

for testCat in testCatNames:
    src = path.join(resourcesPath, testCat)
    dst = path.join(testPath + '/cat', testCat)
    shutil.copyfile(src, dst)

for testDog in testDogNames:
    src = path.join(resourcesPath, testDog)
    dst = path.join(testPath + '/dog', testDog)
    shutil.copyfile(src, dst)


for trainCat in ['cat.{}.jpg'.format(i) for i in range(6000)]:
    src = path.join(resourcesPath, trainCat)
    dst = path.join(fullPath + '/train/cat', trainCat)
    shutil.copyfile(src, dst)

for trainDog in ['dog.{}.jpg'.format(i) for i in range(6000)]:
    src = path.join(resourcesPath, trainDog)
    dst = path.join(fullPath + '/train/dog', trainDog)
    shutil.copyfile(src, dst)

for trainCat in ['cat.{}.jpg'.format(i) for i in range(6000, 10000)]:
    src = path.join(resourcesPath, trainCat)
    dst = path.join(fullPath + '/validation/cat', trainCat)
    shutil.copyfile(src, dst)

for trainDog in ['dog.{}.jpg'.format(i) for i in range(6000, 10000)]:
    src = path.join(resourcesPath, trainDog)
    dst = path.join(fullPath + '/validation/dog', trainDog)
    shutil.copyfile(src, dst)

for trainCat in ['cat.{}.jpg'.format(i) for i in range(10000, 12500)]:
    src = path.join(resourcesPath, trainCat)
    dst = path.join(fullPath + '/test/cat', trainCat)
    shutil.copyfile(src, dst)

for trainDog in ['dog.{}.jpg'.format(i) for i in range(10000, 12500)]:
    src = path.join(resourcesPath, trainDog)
    dst = path.join(fullPath + '/test/dog', trainDog)
    shutil.copyfile(src, dst)